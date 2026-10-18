"""Meaningful timing / output checks; run with python -m unittest discover -s ."""
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from core import (compile_plan, default_policy, inspect_media, make_demo, preview_frame,
                  render_video, run_ffmpeg, validate_events)
from detectors import matches, scan_srt


class PolicyTests(unittest.TestCase):
    def test_overlap_resolves_visual_and_audio_independently(self):
        events = [{"start": 0, "end": 3, "category": "violence"},
                  {"start": 2, "end": 4, "category": "sexual_activity"},
                  {"start": 1, "end": 5, "category": "language"}]
        policy = default_policy()
        policy.update(pre_padding=0, post_padding=0)
        plan = compile_plan(events, policy, 6)
        at = lambda t: next(e for e in plan if e["start"] <= t < e["end"])
        self.assertEqual((at(1.5)["visual"], at(1.5)["audio"]), ("blur", "beep"))
        self.assertEqual((at(2.5)["visual"], at(2.5)["audio"]), ("hide", "mute"))
        self.assertEqual((at(4.5)["visual"], at(4.5)["audio"]), ("allow", "beep"))

    def test_padding_clamped_and_disabled_or_low_severity_excluded(self):
        policy = default_policy()
        policy["categories"]["violence"]["min_severity"] = 2
        events = [{"start": .1, "end": .9, "category": "violence", "severity": 1},
                  {"start": .1, "end": .9, "category": "language", "enabled": False},
                  {"start": .1, "end": .9, "category": "sexual_activity"}]
        self.assertEqual([(e["start"], e["end"]) for e in compile_plan(events, policy, 1)], [(0, 1)])

    def test_invalid_input_fails(self):
        for event in [{"start": float("nan"), "end": 1, "category": "nudity"},
                      {"start": 2, "end": 1, "category": "nudity"},
                      {"start": 0, "end": 1, "category": "unknown"},
                      {"start": 0, "end": 1, "category": "nudity", "enabled": "false"}]:
            with self.assertRaises((ValueError, TypeError)):
                validate_events([event], 3)

    def test_subtitle_matching_uses_words_and_clips_duration(self):
        self.assertFalse(matches("Scunthorpe classic", ["cunt", "ass"]))
        self.assertTrue(matches("A BAD, word here", ["bad word"]))
        events = scan_srt("1\n00:00:01,000 --> 00:00:05,000\nThe banana is ready.\n", ["banana"], 3)
        self.assertEqual([(e["start"], e["end"]) for e in events], [(1, 3)])

    def test_region_does_not_conceal_rest_of_frame_and_frame_hide_wins(self):
        p=default_policy();p.update(pre_padding=0,post_padding=0)
        events=[dict(start=1,end=3,category='nudity',region=[.1,.2,.3,.4]),dict(start=2,end=4,category='sexual_activity')]
        plan=compile_plan(events,p,5)
        self.assertEqual(plan[0]['visual'],'allow')
        self.assertEqual(plan[0]['regions'][0]['box'],[.1,.2,.3,.4])
        self.assertEqual(plan[1]['visual'],'hide')
        self.assertNotIn('regions',plan[1])
        for box in ([0,0,0,1],[.8,0,.3,.2],[float('nan'),0,.2,.2]):
            with self.assertRaises(ValueError):validate_events([dict(start=0,end=1,category='nudity',region=box)],2)


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="censor-tests-")
        cls.source = Path(cls.temp.name) / "original.mp4"
        cls.output = Path(cls.temp.name) / "filtered.mp4"
        events = make_demo(cls.source)
        cls.plan = compile_plan(events, default_policy(), 12)
        render_video(cls.source, cls.output, cls.plan)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def frame(self, path, t):
        return np.asarray(Image.open(io.BytesIO(preview_frame(path, t, [])))).astype(float)

    def audio(self, path, t, length=.25):
        raw = run_ffmpeg(["-ss", str(t), "-i", str(path), "-t", str(length), "-vn",
                          "-ac", "1", "-ar", "48000", "-f", "f32le", "-"]).stdout
        return np.frombuffer(raw, dtype=np.float32)

    def test_hidden_frame_is_opaque_and_clean_frame_remains(self):
        self.assertLess(self.frame(self.output, 6).max(), 3)
        self.assertLess(np.abs(self.frame(self.source, 1) - self.frame(self.output, 1)).mean(), 8)

    def test_blur_removes_high_frequency_detail(self):
        original, blurred = self.frame(self.source, 3), self.frame(self.output, 3)
        detail = lambda frame: np.abs(np.diff(frame, axis=1)).mean()
        self.assertLess(detail(blurred), .55 * detail(original))

    def test_audio_is_muted_or_replaced_and_duration_is_preserved(self):
        self.assertLess(np.sqrt(np.mean(self.audio(self.output, 6) ** 2)), .001)
        self.assertGreater(np.sqrt(np.mean(self.audio(self.output, 1) ** 2)), .01)
        audio = self.audio(self.output, 9.2)
        frequency = np.fft.rfftfreq(len(audio), 1 / 48000)[np.argmax(np.abs(np.fft.rfft(audio)))]
        self.assertAlmostEqual(frequency, 880, delta=8)
        self.assertAlmostEqual(inspect_media(self.output)["duration"], 12, delta=.1)

    def test_video_without_audio_renders(self):
        silent = Path(self.temp.name) / "silent.mp4"
        output = Path(self.temp.name) / "silent-filtered.mp4"
        run_ffmpeg(["-y", "-i", str(self.source), "-an", "-c:v", "copy", str(silent)])
        render_video(silent, output, self.plan)
        self.assertFalse(inspect_media(output)["has_audio"])

    def test_localized_export_preserves_pixels_outside_mask(self):
        output=Path(self.temp.name)/'regions.mp4'
        p=default_policy();p.update(pre_padding=0,post_padding=0)
        events=[dict(start=2,end=4,category='nudity',region=[.2,.2,.3,.4]),
                dict(start=5,end=7,category='violence',source='manual',region=[.2,.2,.3,.4])]
        render_video(self.source,output,compile_plan(events,p,12))
        original=self.frame(self.source,3);filtered=self.frame(output,3)
        self.assertLess(filtered[90:200,140:300].max(),5)
        self.assertLess(np.abs(original[:,400:]-filtered[:,400:]).mean(),8)

        original=self.frame(self.source,6);filtered=self.frame(output,6)
        # Squared gradients distinguish sharp edges from the same contrast spread
        # across a smooth blur; total absolute contrast need not decrease.
        detail=lambda a:np.square(np.diff(a[90:200,145:300],axis=1)).mean()
        self.assertLess(detail(filtered),.6*detail(original))
        self.assertLess(np.abs(original[:,400:]-filtered[:,400:]).mean(),8)

    def test_shot_padding_does_not_conceal_clean_pixels_after_cut(self):
        output=Path(self.temp.name)/'shot-boundary.mp4'
        event=dict(start=.7,end=1,category='nudity',time_bounds=[0,1])
        render_video(self.source,output,compile_plan([event],default_policy(),12))
        self.assertLess(self.frame(output,.9).max(),5)
        clean=self.frame(output,1.1);original=self.frame(self.source,1.1)
        self.assertGreater(clean.std(),30)
        self.assertLess(np.abs(clean-original).mean(),8)


if __name__ == "__main__":
    unittest.main()
