"""Fine-tuning loop on a tiny stand-in backbone: masking, held-out titles, thresholds, apply."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from labeling import LABEL_CATEGORIES, write_json

AVAILABLE = importlib.util.find_spec('torch') and importlib.util.find_spec('sklearn')


def tiny_backbone():
    import torch

    class Tiny(torch.nn.Module):
        """Tokens carry the clip's mean brightness, like a feature a real backbone could learn."""
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=16)
            self.embed = torch.nn.Linear(1, 16)
            self.encoder = torch.nn.Module()
            self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(2)])
            self.encoder.layernorm = torch.nn.LayerNorm(16)

        def get_vision_features(self, videos):
            hidden = self.embed(videos.mean(dim=(1, 2, 3, 4))[:, None, None].expand(-1, 4, 1))
            for layer in self.encoder.layer:
                hidden = hidden + layer(hidden)
            return self.encoder.layernorm(hidden)

    torch.manual_seed(0)  # frozen layers must match between training and apply, like pretrained weights
    return Tiny()


def synthetic_clips(folder, window):
    centers = np.arange(1.0, 15.5 + 1e-9, .5)
    bright = (centers >= 4) & (centers < 8) & window['positive']
    clips = np.zeros((len(centers), 2, 4, 4, 3), dtype=np.uint8)
    clips[bright] = 200
    clips[~bright] = np.random.default_rng(int(window['id'][1:])).integers(0, 60, size=(int((~bright).sum()), 2, 4, 4, 3))
    return centers, clips


@unittest.skipUnless(AVAILABLE, 'fine-tuning needs the .train-venv environment')
class FinetuneTests(unittest.TestCase):
    def test_probe_learns_a_signal_and_calibrates_on_held_out_titles(self):
        import finetune
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'labeling' / 'train-x'
            windows = []
            for i in range(20):
                window = dict(id=f'w{i}', title=f'title {i % 10}', split='train', target=[2.0, 14.0], clip_duration=16.0,
                              positive=bool(i % 2))
                windows.append(window)
                categories = {c: dict(status='reviewed', intervals=[], ambiguous=[]) for c in LABEL_CATEGORIES}
                if window['positive']:
                    categories['violence']['intervals'] = [[4, 8]]
                categories['kissing']['status'] = 'unsure'
                write_json(folder / 'labels' / f'w{i}.json', dict(categories=categories, drafts_shown=False))
            write_json(folder / 'project.json', dict(name='train-x', windows=windows))
            with patch('finetune.window_clips', synthetic_clips):
                state, meta = finetune.train(Path(directory), [folder], unfreeze=1, epochs=25, device='cpu',
                                             backbone=tiny_backbone(), log=lambda *_: None)
                self.assertEqual(len(meta['held_out_titles']), 2)
                self.assertFalse(set(meta['held_out_titles']) & set(meta['train_titles']))
                self.assertGreater(meta['categories']['violence']['held_out_ap'], .9)
                self.assertIn('skipped', meta['categories']['gore'])
                self.assertTrue(any(k.startswith('backbone.encoder.layer.1.') for k in state))
                self.assertFalse(any(k.startswith('backbone.encoder.layer.0.') for k in state))
                model, _ = finetune.build_model(Path(directory), 1, tiny_backbone())
                model.load_state_dict(state, strict=False)
                entries, events = finetune.apply(Path(directory), folder, windows[1], model.eval(), meta, device='cpu')
            self.assertTrue(entries and all(e['category'] == 'violence' for e in entries))
            self.assertTrue(all(3.5 <= e['sample_time'] < 8.5 for e in entries))


if __name__ == '__main__':
    unittest.main()
