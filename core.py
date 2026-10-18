"""Policy compilation and real MP4 rendering. Times are seconds, [start, end)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from collections import Counter

import imageio_ffmpeg

CATEGORIES = ("nudity", "sexual_activity", "violence", "gore", "kissing", "language", "custom")
VISUAL = ("allow", "blur", "hide")
AUDIO = ("allow", "beep", "mute")


def run_ffmpeg(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", *args],
        capture_output=True, timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-5000:])
    return result


def inspect_media(path: str | Path) -> dict:
    result = run_ffmpeg(["-i", str(path), "-t", "0", "-f", "null", "-"])
    report = result.stderr.decode("utf-8", "replace")
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", report)
    if not match or not re.search(r"Stream #0:\d+.*Video:", report):
        raise ValueError("Use a finite video file with a readable duration.")
    hours, minutes, seconds = map(float, match.groups())
    return {"duration": hours * 3600 + minutes * 60 + seconds,
            "has_audio": bool(re.search(r"Stream #0:\d+.*Audio:", report))}


def number(value, low: float, high: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return value


def validate_events(events: list[dict], duration: float) -> list[dict]:
    if not isinstance(events, list) or len(events) > 250000:
        raise ValueError("The timeline must be a list of at most 250,000 events.")
    clean = []
    for event in events:
        if event.get("category") not in CATEGORIES:
            raise ValueError("Unknown event category.")
        start = number(event.get("start"), 0, duration, "Start")
        end = number(event.get("end"), 0, duration, "End")
        if start >= end:
            raise ValueError("Every event must end after it starts.")
        confidence = number(event.get("confidence", 1), 0, 1, "Detector score")
        severity = number(event.get("severity", 2), 1, 3, "Severity")
        if severity not in (1, 2, 3):
            raise ValueError("Severity must be 1, 2, or 3.")
        enabled = event.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("Event enabled must be true or false.")
        review_required = event.get('review_required', False)
        confirmed = event.get('confirmed', False)
        if type(review_required) is not bool or type(confirmed) is not bool:
            raise ValueError('Review flags must be true or false.')
        region = event.get("region")
        bounds = event.get('time_bounds')
        if bounds is not None:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError('Time bounds need a shot start and end')
            bounds = [number(v, 0, duration, 'Shot bound') for v in bounds]
            if not bounds[0] <= start < end <= bounds[1]:
                raise ValueError('Event must lie within its shot bounds')
        if region is not None:
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                raise ValueError("A region needs normalized x, y, width and height.")
            region = [number(v, 0, 1, "Region coordinate") for v in region]
            if region[2] <= 0 or region[3] <= 0 or region[0]+region[2] > 1.000001 or region[1]+region[3] > 1.000001:
                raise ValueError("The region must fit inside the picture.")
        clean.append({"start": start, "end": end, "category": event["category"],
                      "confidence": confidence, "severity": severity, "enabled": enabled,
                      "review_required": review_required, "confirmed": confirmed,
                      "source": str(event.get("source", "manual"))[:100],
                      "note": str(event.get("note", ""))[:300], **({"region": region} if region else {}),
                      **({'time_bounds': bounds} if bounds is not None else {})})
    return sorted(clean, key=lambda e: (e["start"], e["end"]))


def default_policy() -> dict:
    rules = {category: {"visual": "hide", "blur": 50, "audio": "allow",
                        "min_severity": 1, "min_confidence": .55}
             for category in CATEGORIES}
    rules["violence"].update(visual="blur", blur=35)
    rules["custom"].update(visual="blur")
    rules["kissing"].update(visual="allow")
    rules["sexual_activity"].update(audio="mute")
    rules["language"].update(visual="allow", audio="beep", min_confidence=0)
    for category in ("gore", "kissing", "custom"):
        rules[category]["min_confidence"] = .70
    rules["sexual_activity"]["min_confidence"] = .65
    rules["violence"]["min_confidence"] = .65
    rules["nudity"]["scope"] = "region"
    return {"pre_padding": .3, "post_padding": .35, "categories": rules}


def compile_plan(events: list[dict], policy: dict, duration: float) -> list[dict]:
    """Resolve overlap independently: hide > blur > allow; mute > beep > allow."""
    events = validate_events(events, duration)
    pre = number(policy.get("pre_padding", 0), 0, 3, "Leading padding")
    post = number(policy.get("post_padding", 0), 0, 3, "Trailing padding")
    candidates = []
    for event in events:
        rule = policy.get("categories", {}).get(event["category"], {})
        visual, audio = rule.get("visual", "allow"), rule.get("audio", "allow")
        if visual not in VISUAL or audio not in AUDIO:
            raise ValueError("Unknown visual or audio action.")
        strength = number(rule.get("blur", 50), 1, 100, "Blur strength")
        severity = number(rule.get("min_severity", 1), 1, 3, "Minimum severity")
        confidence = number(rule.get("min_confidence", 0), 0, 1, "Minimum score")
        if not event["enabled"] or event["severity"] < severity:
            continue
        if (event['review_required'] or event['source'].startswith('CLIP')) and not event['confirmed']:
            continue
        if event["confidence"] < confidence:
            continue
        if visual == "allow" and audio == "allow":
            continue
        scope = rule.get("scope", "frame")
        if scope not in ("frame", "region"):
            raise ValueError("Unknown filter area.")
        region = event.get("region") if scope == "region" or event["source"] == "manual" else None
        low, high = event.get('time_bounds', [0, duration])
        candidates.append({"start": max(0, low, event["start"] - pre),
                           "end": min(duration, high, event["end"] + post),
                           "visual": visual, "blur": strength if visual == "blur" else 0,
                           "audio": audio, "region": tuple(region) if region and visual != "allow" else None})
    changes = {}
    for event in candidates:
        key = (event["visual"], event["blur"], event["audio"], event["region"])
        changes.setdefault(event["start"], []).append((key, 1))
        changes.setdefault(event["end"], []).append((key, -1))
    boundaries = sorted(changes)
    active = Counter()
    plan = []
    for start, end in zip(boundaries, boundaries[1:]):
        for key, delta in changes[start]:
            active[key] += delta
            if not active[key]:
                del active[key]
        if not active:
            continue
        frame_actions = [e for e in active if e[3] is None]
        visual = max(frame_actions, key=lambda e: (VISUAL.index(e[0]), e[1]), default=("allow", 0))
        audio = max((e[2] for e in active), key=AUDIO.index)
        item = {"start": start, "end": end, "visual": visual[0],
                "blur": visual[1], "audio": audio}
        regions = sorted({(e[3], e[0], e[1]) for e in active if e[3] is not None})
        if regions and item["visual"] != "hide":
            item["regions"] = [dict(box=list(box), visual=v, blur=b) for box, v, b in regions]
        if (plan and abs(plan[-1]["end"] - start) < 1e-8
                and all(plan[-1][key] == item[key] for key in ("visual", "blur", "audio"))
                and plan[-1].get("regions", []) == item.get("regions", [])):
            plan[-1]["end"] = end
        else:
            plan.append(item)
    return plan


def video_graph(plan):
    graph = ["[0:v:0]scale=trunc(iw/2)*2:trunc(ih/2)*2[v0]"]
    current = "v0"
    index = 0
    for event in plan:
        condition = f"gte(t,{event['start']:.6f})*lt(t,{event['end']:.6f})"
        operations = [(event['visual'], event.get('blur', 50), None)]
        # Apply opaque regions last where masks overlap.
        operations += [(r['visual'], r['blur'], r['box']) for r in sorted(event.get('regions', []), key=lambda r: VISUAL.index(r['visual']))]
        for visual, strength, box in operations:
            if visual == 'allow':
                continue
            index += 1
            out = f"v{index}"
            if box:
                x, y, w, h = box
                geometry = f"x=iw*{x:.8f}:y=ih*{y:.8f}:w=iw*{w:.8f}:h=ih*{h:.8f}"
                if visual == 'hide':
                    graph.append(f"[{current}]drawbox={geometry}:color=black:t=fill:enable='{condition}'[{out}]")
                else:
                    graph += [f"[{current}]split[base{index}][patch{index}]",
                              f"[patch{index}]crop=w=max(2\\,iw*{w:.8f}):h=max(2\\,ih*{h:.8f}):x=iw*{x:.8f}:y=ih*{y:.8f},gblur=sigma={2+strength*.78:.3f}:steps=2[blur{index}]",
                              f"[base{index}][blur{index}]overlay=x=main_w*{x:.8f}:y=main_h*{y:.8f}:eof_action=pass:enable='{condition}'[{out}]"]
            else:
                effect = "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill" if visual == 'hide' else f"gblur=sigma={2+strength*.78:.3f}:steps=2"
                graph.append(f"[{current}]{effect}:enable='{condition}'[{out}]")
            current = out
    graph.append(f"[{current}]null[v]")
    return ';'.join(graph)


def render_video(source: Path, destination: Path, plan: list[dict]) -> dict:
    if source.resolve() == destination.resolve():
        raise ValueError("Output must differ from the source video.")
    info = inspect_media(source)
    audio_filters = []
    for event in plan:
        start, end = event["start"], event["end"]
        condition = f"gte(t,{start:.6f})*lt(t,{end:.6f})"
        if event["audio"] != "allow":
            replacement = "0" if event["audio"] == "mute" else "0.08*sin(2*PI*880*t)"
            audio_filters.append(f"aeval=exprs='if({condition},{replacement},val(ch))':c=same")
    graph = video_graph(plan)
    if info["has_audio"]:
        graph += f";[0:a:0]{','.join(audio_filters) if audio_filters else 'anull'}[a]"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render-", dir=destination.parent) as temp:
        graph_file = Path(temp) / "filters.txt"
        graph_file.write_text(graph, encoding="utf-8")
        temporary_output = Path(temp) / "filtered.mp4"
        args = ["-y", "-i", str(source), "-filter_complex_threads", "2", "-filter_complex_script", str(graph_file), "-map", "[v]"]
        if info["has_audio"]:
            args += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
                 "-map_metadata", "-1", "-movflags", "+faststart", str(temporary_output)]
        run_ffmpeg(args)
        temporary_output.replace(destination)
    return info


def preview_frame(source: Path, time: float, plan: list[dict]) -> bytes:
    event = next((e for e in plan if e["start"] <= time < e["end"]), None)
    graph = video_graph([dict(event, start=0, end=2)] if event else []) + ';[v]scale=640:-2[out]'
    return run_ffmpeg(["-ss", str(time), "-i", str(source), "-frames:v", "1", "-filter_complex_threads", "2", "-filter_complex", graph, '-map', '[out]',
                       "-f", "image2pipe", "-vcodec", "png", "-"]).stdout


def make_demo(path: Path) -> list[dict]:
    """Harmless generated test pattern. Labels below are scripted, not AI detections."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        run_ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=12",
                    "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=12",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-movflags", "+faststart", str(path)])
    return [{"start": start, "end": end, "category": category, "confidence": 1,
             "severity": 2, "enabled": True, "source": "scripted demo",
             "note": "Synthetic test pattern; this is a preset event, not AI detection."}
            for start, end, category in [(2, 4, "violence"), (5, 7, "sexual_activity"), (9, 10, "language")]]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render a reviewed media-filter timeline.")
    parser.add_argument("source", type=Path)
    parser.add_argument("timeline", type=Path, help="JSON containing events and policy")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    document = json.loads(args.timeline.read_text(encoding="utf-8"))
    plan = compile_plan(document["events"], document["policy"], inspect_media(args.source)["duration"])
    render_video(args.source, args.output, plan)
    print(f"Rendered {args.output}")
