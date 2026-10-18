"""AI labeling support: contact sheets for viewing windows, and writing labels with provenance.

Labels written here record labeler 'claude' and that no model drafts were seen.
Reports keep them apart from human labels; a human-checked sample measures
their agreement. Times on sheets and in decision files are seconds from the
start of the labeled target, so 0-12 for a 12-second window.

    python ai_label.py sheets --name test-1 --out sheets/
    python ai_label.py zoom --name test-1 --id castle-s07e06-26 --start 2 --end 5 --out sheets/
    python ai_label.py write --name test-1 --decisions decisions.json
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from analysis_engine import native_frames
from labeling import LABEL_CATEGORIES, project_dir, read_json, validate_labels, write_json
from model_setup import prepare_environment


def sheet(frames, low, columns, tile, title):
    rows = (len(frames) + columns - 1) // columns
    width, height = tile
    image = Image.new('RGB', (columns * width, rows * (height + 18) + 22), '#101418')
    draw = ImageDraw.Draw(image)
    draw.text((6, 4), title, fill='#bce899')
    for i, (t, pixels) in enumerate(frames):
        picture = Image.fromarray(pixels).resize((width, height))
        x, y = (i % columns) * width, 22 + (i // columns) * (height + 18)
        image.paste(picture, (x, y + 18))
        draw.text((x + 4, y + 3), f'{t - low:6.2f}s', fill='white')
    return image


def window_sheets(folder: Path, window: dict, out: Path, fps=4, per_sheet=24, columns=6, tile=(320, 180)):
    low, high = window['target']
    frames = native_frames(folder / 'clips' / f"{window['id']}.mp4", low, high, fps)
    paths = []
    for part in range(0, len(frames), per_sheet):
        path = out / f"{window['id']}-{part // per_sheet}.jpg"
        sheet(frames[part:part + per_sheet], low, columns, tile,
              f"{window['id']} · {window['title']} {window['episode']} · sheet {part // per_sheet + 1}").save(path, quality=88)
        paths.append(path)
    return paths


def to_clip(window, spans):
    low = window['target'][0]
    return [[low + a, low + b] for a, b in spans]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sheets = sub.add_parser('sheets'); sheets.add_argument('--name', required=True); sheets.add_argument('--out', type=Path, required=True)
    sheets.add_argument('--ids', nargs='*'); sheets.add_argument('--fps', type=float, default=4)
    sheets.add_argument('--compact', action='store_true', help='One sheet per window: 48 smaller tiles')
    zoom = sub.add_parser('zoom'); zoom.add_argument('--name', required=True); zoom.add_argument('--id', required=True)
    zoom.add_argument('--start', type=float, required=True); zoom.add_argument('--end', type=float, required=True)
    zoom.add_argument('--fps', type=float, default=12); zoom.add_argument('--out', type=Path, required=True)
    write = sub.add_parser('write'); write.add_argument('--name', required=True); write.add_argument('--decisions', type=Path, required=True)
    args = parser.parse_args()
    folder = project_dir(prepare_environment(), args.name)
    windows = {w['id']: w for w in read_json(folder / 'project.json')['windows']}

    if args.command == 'sheets':
        args.out.mkdir(parents=True, exist_ok=True)
        for window_id in args.ids or windows:
            layout = dict(per_sheet=48, columns=8, tile=(240, 135)) if args.compact else {}
            for path in window_sheets(folder, windows[window_id], args.out, args.fps, **layout):
                print(path)
    elif args.command == 'zoom':
        args.out.mkdir(parents=True, exist_ok=True)
        window = windows[args.id]
        low = window['target'][0]
        frames = native_frames(folder / 'clips' / f'{args.id}.mp4', low + args.start, low + args.end, args.fps)
        path = args.out / f'{args.id}-zoom-{args.start:g}-{args.end:g}.jpg'
        sheet(frames[:24], low, 4, (480, 270), f'{args.id} zoom {args.start:g}-{args.end:g}s').save(path, quality=90)
        print(path)
    else:
        decisions = json.loads(args.decisions.read_text(encoding='utf-8'))
        for window_id, decision in decisions.items():
            window = windows[window_id]
            categories = {}
            for category in LABEL_CATEGORIES:
                value = decision.get(category, {})
                if isinstance(value, list):
                    value = dict(intervals=value)
                categories[category] = dict(status=value.get('status', 'reviewed'),
                                            intervals=to_clip(window, value.get('intervals', [])),
                                            ambiguous=to_clip(window, value.get('ambiguous', [])))
            labels = validate_labels(window, dict(categories=categories, drafts_shown=False, labeler='claude',
                                                  note=decision.get('note', '')))
            write_json(folder / 'labels' / f'{window_id}.json', labels)
        print(f'Wrote {len(decisions)} AI-labeled windows')


if __name__ == '__main__':
    main()
