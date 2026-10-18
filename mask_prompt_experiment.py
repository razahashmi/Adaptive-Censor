"""Training-selected, paired SAM prompt experiment; never edits human labels.

This is an offline geometry experiment, not a semantic detector or playback
qualification. Select the prompt using training masks before scoring regression
labels. Saved JPEGs are replayed identically for all candidates; old drafts were
made before JPEG encoding and are not the paired baseline.
"""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image

from evaluate_masks import counts, rates
from labeling import read_json, write_json
from provenance import digest, file_hash

MODES = ('box', 'box-best', 'center', 'center-best')
PROJECT = 'mask-{}-batch-more-20261012-2'


def envelope(shape, box):
    h, w = shape
    x1, y1, x2, y2 = box
    mx, my = (x2-x1)*.1, (y2-y1)*.1
    result = np.zeros(shape, dtype=bool)
    result[max(0, int(y1-my)):min(h, int(np.ceil(y2+my))),
           max(0, int(x1-mx)):min(w, int(np.ceil(x2+mx)))] = True
    return result


def combine(masks, scores, boxes, shape):
    """Choose each prompt's own best mask, then clip to its own envelope."""
    masks, scores = np.asarray(masks), np.asarray(scores)
    if masks.ndim != 4 or masks.shape[0] != len(boxes) or masks.shape[2:] != shape:
        raise ValueError('Prompt/mask dimensions differ')
    if scores.shape != masks.shape[:2] or not np.isfinite(scores).all():
        raise ValueError('Prompt/mask quality scores differ')
    result = np.zeros(shape, dtype=bool)
    for i, box in enumerate(boxes):
        result |= masks[i, scores[i].argmax()].astype(bool) & envelope(shape, box)
    return result


def variants(drafter, pixels, boxes):
    """Four prompt choices share a single image encoder pass."""
    shape = pixels.shape[:2]
    if not boxes:
        return {mode: np.zeros(shape, dtype=bool) for mode in MODES}, {}
    torch = drafter.torch
    processor, model = drafter.processor, drafter.model
    started = time.perf_counter()
    base = processor(images=Image.fromarray(pixels), input_boxes=[boxes], return_tensors='pt').to(drafter.device)
    if drafter.device == 'cuda':
        torch.cuda.synchronize()
    with torch.inference_mode():
        embeddings = model.get_image_embeddings(base['pixel_values'])
        if drafter.device == 'cuda':
            torch.cuda.synchronize()
        encoded = time.perf_counter()
        masks, timings = {}, {'encode_seconds': encoded-started}
        for center in (False, True):
            prompts = dict(input_boxes=base['input_boxes'])
            if center:
                points = [[[(x1+x2)/2, (y1+y2)/2]] for x1, y1, x2, y2 in boxes]
                additional = processor(original_sizes=base['original_sizes'], input_points=[points],
                                       input_labels=[[[1] for _ in boxes]], return_tensors='pt').to(drafter.device)
                prompts.update(input_points=additional['input_points'], input_labels=additional['input_labels'])
            for multi in (False, True):
                name = ('center' if center else 'box') + ('-best' if multi else '')
                t = time.perf_counter()
                output = model(image_embeddings=embeddings, **prompts, multimask_output=multi)
                native = processor.post_process_masks(output.pred_masks.cpu(), base['original_sizes'])[0].numpy()
                masks[name] = combine(native, output.iou_scores[0].float().cpu().numpy(), boxes, shape)
                timings[name] = time.perf_counter()-t
    return masks, timings


def input_contract(folder):
    from segmentation_labels import read_label
    data = read_json(folder/'dataset.json')
    bound = {'dataset.json': file_hash(folder/'dataset.json')}
    rows = []
    for row in data['rows']:
        label = read_label(folder, data, row)
        if label is None:
            continue
        if label['labeler'] != 'human':
            raise ValueError('Human labels required')
        if label['status'] != 'reviewed':
            continue
        for suffix in ('.jpg', '-draft.png', '-gold.png'):
            name = row['id']+suffix
            bound[name] = file_hash(folder/name)
        name = 'labels/'+row['id']+'.json'
        bound[name] = file_hash(folder/name)
        exposure = 'exposure/'+row['id']+'.json'
        if (folder/exposure).exists():
            bound[exposure] = file_hash(folder/exposure)
        rows.append(row['id'])
    return dict(project=data['name'], split=data['split'], inputs=bound, rows=rows)


def verify(folder, contract):
    for name, sha in contract['inputs'].items():
        if file_hash(folder/name) != sha:
            raise ValueError('Frozen mask input changed: '+name)


def predictions(root, destination, split, modes=MODES):
    from segmentation_data import MaskDrafter
    folder = root/'segmentation'/PROJECT.format(split)
    data = read_json(folder/'dataset.json')
    drafter = None
    for n, row in enumerate(data['rows']):
        target = destination/'predictions'/split/row['id']
        identity = dict(image_sha256=file_hash(folder/(row['id']+'.jpg')),
                        boxes=row['prompt_boxes'], code_sha256=file_hash(Path(__file__)))
        if identity['image_sha256'] != row['image_sha256']:
            raise ValueError('Source frame changed')
        if (target/'done.json').exists():
            if read_json(target/'done.json')['identity'] != identity:
                raise ValueError('Stale prompt prediction')
            for mode in modes:
                if file_hash(target/(mode+'.png')) != read_json(target/'done.json')['outputs'][mode]:
                    raise ValueError('Prediction mask changed')
            continue
        if drafter is None:
            drafter = MaskDrafter(root)
        with Image.open(folder/(row['id']+'.jpg')) as image:
            pixels = np.asarray(image.convert('RGB'))
        boxes = [hit['box'] for hit in row['prompt_boxes']]
        output, timings = variants(drafter, pixels, boxes)
        target.mkdir(parents=True, exist_ok=True)
        for mode, mask in output.items():
            Image.fromarray(mask.astype(np.uint8)*255).save(target/(mode+'.png'))
        write_json(target/'done.json', dict(identity=identity, timings=timings,
                   outputs={mode:file_hash(target/(mode+'.png')) for mode in MODES},
                   model_identity=drafter.identity))
        if n % 10 == 0 or n+1 == len(data['rows']):
            print(split, n+1, '/', len(data['rows']), flush=True)


def score(root, destination, split, contract, modes=MODES, blind=False):
    from segmentation_labels import read_label
    folder = root/'segmentation'/PROJECT.format(split)
    verify(folder, contract)
    data = read_json(folder/'dataset.json')
    labels = {r['id']:read_label(folder, data, r) for r in data['rows']}
    blocked = {(r['title'],r['group']) for r in data['rows']
               if labels[r['id']] and labels[r['id']]['drafts_shown']} if blind else set()
    results = {}
    for mode in modes:
        total = dict(tp=0, fp=0, fn=0, clean=0, gold=0)
        samples = []
        for row in data['rows']:
            if row['id'] not in contract['rows'] or (row['title'],row['group']) in blocked:
                continue
            with Image.open(folder/(row['id']+'-gold.png')) as image:
                gold = np.asarray(image.convert('L')) >= 128
            with Image.open(destination/'predictions'/split/row['id']/(mode+'.png')) as image:
                predicted = np.asarray(image.convert('L')) >= 128
            c = counts(predicted, gold)
            for k,v in c.items():
                total[k] += v
            samples.append(dict(id=row['id'], title=row['title'], group=row['group'], counts=c))
        clean = [r for r in samples if not r['counts']['gold']]
        results[mode] = dict(counts=total, **rates(total), frames=len(samples),
            positive_groups=len({(r['title'],r['group']) for r in samples if r['counts']['gold']}),
            positive_titles=len({r['title'] for r in samples if r['counts']['gold']}),
            clean_frames=len(clean), false_clean_frames=sum(bool(r['counts']['fp']) for r in clean), samples=samples)
    verify(folder, contract)
    return results


def run(root, destination, stage):
    destination.mkdir(parents=True, exist_ok=True)
    if stage == 'train':
        folder = root/'segmentation'/PROJECT.format('train')
        contract = input_contract(folder)
        setup = dict(training=contract, candidates=list(MODES), code_sha256=file_hash(Path(__file__)),
            selection='Highest training pixel IoU with coverage >= paired baseline and no extra clean frames falsely masked',
            scope='Previously inspected title-separated development geometry; not event-recognition accuracy or qualification')
        old = read_json(destination/'experiment.json', {})
        if old and old != setup:
            raise ValueError('Experiment changed; use a new output directory')
        write_json(destination/'experiment.json', setup)
        predictions(root, destination, 'train')
        results = score(root, destination, 'train', contract)
        base = results['box']
        eligible = [m for m in MODES if results[m]['coverage'] >= base['coverage']
                    and results[m]['false_clean_frames'] <= base['false_clean_frames']]
        choice = max(eligible, key=lambda m: results[m]['iou'])
        write_json(destination/'training.json', results)
        write_json(destination/'choice.json', dict(mode=choice, experiment_identity=digest(setup),
            training_report_sha256=file_hash(destination/'training.json'), selected_before_regression=True))
        print('Training choice:', choice, flush=True)
    else:
        setup = read_json(destination/'experiment.json')
        choice = read_json(destination/'choice.json')
        if choice['experiment_identity'] != digest(setup) or file_hash(Path(__file__)) != setup['code_sha256']:
            raise ValueError('Experiment/selection changed')
        verify(root/'segmentation'/PROJECT.format('train'), setup['training'])
        if file_hash(destination/'training.json') != choice['training_report_sha256']:
            raise ValueError('Training selection changed')
        for split in ('val','test'):
            predictions(root, destination, split)
            contract = input_contract(root/'segmentation'/PROJECT.format(split))
            results = score(root, destination, split, contract, modes=('box', choice['mode']), blind=True)
            write_json(destination/(split+'.json'), dict(contract=contract, choice=choice, results=results,
                qualification_passed=False, evaluation_scope=setup['scope']))
    print('Finished', stage, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('train','evaluate'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from model_setup import prepare_environment
    run(prepare_environment(), a.output, a.stage)
