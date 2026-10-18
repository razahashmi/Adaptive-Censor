"""Bounded, incremental analysis. Coverage is committed only after every enabled pass succeeds."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import hashlib
import json
import math
import os
import threading
import time
import uuid

import av
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from core import run_ffmpeg
from detectors import tokens, NUDITY_CLASSES
from model_setup import prepare_environment, speech_model_path
from scene_verifier import VERIFIER_REVISION
from provenance import digest, file_hash, head_identity, pipeline_identity, qualified, qualification_identity
from frame_cache import BoundedCache, source_key

_decode_cache = BoundedCache(128 * 1024**2)
_frame_tiles = BoundedCache(256 * 1024**2)


def _cached_decode(function):
    from functools import wraps
    @wraps(function)
    def wrapped(path, *args):
        key = (function.__name__, source_key(path), tuple(args))
        return _decode_cache.get(key, lambda: function(path, *args))
    return wrapped

CHUNK = 6.0
CONTEXT = 2.0
ANALYSIS_REVISION = 13  # Exact-player training inputs, attentive heads and rejection-capable confirmation.
SCENE_PROMPTS = {
    "sexual_activity": ["a movie scene showing people having sex", "a sexually explicit scene"],
    "violence": ["a violent fight with people punching each other", "a violent attack with a weapon"],
    "gore": ["graphic gore and severe bloody injuries", "a mutilated body with exposed organs"],
    "kissing": ["two people kissing on the lips", "a romantic close-up kiss"],
}
NEUTRAL_PROMPTS = ["an ordinary peaceful scene", "people having a conversation",
                   "a landscape or a room", "a person doing an everyday activity",
                   "text, graphics or animation on a screen", "people playing sports"]


class ScanConfig(BaseModel):
    speech: bool = True
    speech_model: Literal["tiny.en", "small.en"] = "tiny.en"
    nudity: bool = True
    anatomy_model: Literal["320n", "640m"] = "320n"
    anatomy_fps: int = Field(default=12, ge=2, le=30)
    anatomy_tracking: bool = True
    anatomy_roi: str = Field(default='', max_length=64, pattern=r'^[a-z0-9-]*$')
    scenes: bool = True
    # verified: scene-model pass on every chunk; learned: trained heads scan, the scene model
    # checks only their uncertain candidates; review: CLIP suggestions for manual review.
    scene_mode: Literal["verified", "learned", "review"] = "verified"
    scene_filtering: Literal["review", "automatic"] = "review"
    scene_model: Literal["4B", "8B"] = "4B"
    phrases: list[str] = Field(default_factory=lambda: ["fuck", "shit", "asshole", "bitch"])
    allow_phrases: list[str] = Field(default_factory=list)
    custom_visual: str = Field(default="", max_length=500)
    fps: int = Field(default=2, ge=1, le=8)
    lookahead: float = Field(default=30.0, ge=6, le=60)
    device: str = "auto"
    learned_heads: str = Field(default="", max_length=64, pattern=r"^[a-z0-9-]*$")  # empty: learned detectors off
    learned_verification: Literal['off', 'selective', 'confirm'] = 'off'

    @model_validator(mode='after')
    def custom_in_learned_mode(self):
        if self.scenes and self.scene_mode == 'learned' and self.custom_visual.strip():
            raise ValueError('Custom visual concepts need Sequence verification or CLIP review mode')
        return self

    @field_validator("device")
    @classmethod
    def valid_device(cls, value):
        if value not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda")
        return value

    @field_validator("phrases", "allow_phrases")
    @classmethod
    def valid_phrases(cls, values):
        if len(values) > 300:
            raise ValueError("Use at most 300 phrases")
        clean = list(dict.fromkeys(v.strip() for v in values if v.strip()))
        if any(len(v) > 120 or len(tokens(v)) > 6 for v in clean):
            raise ValueError("Each phrase must have at most 6 words and 120 characters")
        return clean


def media_info(path: Path):
    with av.open(str(path)) as container:
        if not container.duration:
            raise ValueError("This source needs a finite, readable duration")
        return {"duration": float(container.duration / av.time_base),
                "has_video": bool(container.streams.video), "has_audio": bool(container.streams.audio),
                "video_codec": container.streams.video[0].codec_context.name if container.streams.video else None,
                "audio_codec": container.streams.audio[0].codec_context.name if container.streams.audio else None}


def normalize_media(source: Path, destination: Path):
    info = media_info(source)
    if not info["has_audio"] and not info["has_video"]:
        raise ValueError("No audio or video stream was found")
    args = ["-y"]
    if not info["has_video"]:
        args += ["-f", "lavfi", "-i", "color=c=0x121c2b:s=960x540:r=10", "-i", str(source),
                 "-map", "0:v:0", "-map", "1:a:0", "-t", str(info["duration"]),
                 "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-c:a", "aac"]
    else:
        args += ["-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
                 "-c:v", "copy" if info["video_codec"] == "h264" else "libx264"]
        if info["video_codec"] != "h264":
            args += ["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        args += ["-c:a", "copy" if info["audio_codec"] == "aac" else "aac"]
    args += ["-sn", "-dn", "-map_metadata", "-1", "-avoid_negative_ts", "make_zero",
             "-movflags", "+faststart", str(destination)]
    run_ffmpeg(args, timeout=7200)
    return media_info(destination)


@_cached_decode
def sample_frames(path: Path, start: float, end: float, fps: int):
    """One seek per chunk, sequential decoding; only downscaled samples survive."""
    _, decoded = decoded_range(path, start, end)
    frames, next_time = [], start
    for t, pixels in decoded:
        if t + 1e-5 < next_time:
            continue
        frames.append((t, pixels))
        next_time = t + 1 / fps - .001
    if not frames:
        raise RuntimeError(f"No decoded frames in interval {start:.2f}–{end:.2f}")
    return frames


@_cached_decode
def audio_samples(path: Path, start: float, end: float):
    """Decode only the requested absolute timestamp window, including fMP4 live fragments."""
    output = np.zeros(round((end - start) * 16000), dtype=np.float32)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return output
        stream = container.streams.audio[0]
        try:
            container.seek(max(0, int(start / float(stream.time_base))), stream=stream, backward=True)
        except av.error.FFmpegError:
            pass
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        def copy(frame):
            if frame.time is None:
                raise RuntimeError("Decoded audio has no timestamp")
            samples = frame.to_ndarray().reshape(-1)
            index = round((float(frame.time) - start) * 16000)
            left, right = max(0, index), min(len(output), index + len(samples))
            if right > left:
                output[left:right] = samples[left-index:right-index]
        for frame in container.decode(stream):
            if frame.time is not None and frame.time > end + .1:
                break
            for converted in resampler.resample(frame):
                copy(converted)
        for converted in resampler.resample(None):
            copy(converted)
    return output


@_cached_decode
def sample_burst(path: Path, center: float):
    """Five distinct native frames about 80ms apart, centered on a candidate.

    Uses decoded timestamps instead of interpolating or duplicating images. At
    23.976fps this is normally every second frame (about 83ms).
    """
    _, decoded = decoded_range(path, max(0, center-.5), center+.5, inclusive=True)
    if not decoded:
        raise RuntimeError('No native frames available for scene evidence')
    middle = min(range(len(decoded)), key=lambda i:abs(decoded[i][0]-center))
    selected = {middle}
    for delta in (-.16, -.08, .08, .16):
        selected.add(min(range(len(decoded)), key=lambda i:abs(decoded[i][0]-(center+delta))))
    for i in sorted(range(len(decoded)), key=lambda i:abs(decoded[i][0]-center)):
        if len(selected) >= min(5,len(decoded)):
            break
        selected.add(i)
    indices = sorted(selected)
    return [decoded[i] for i in indices], indices.index(middle)


def box_iou(first, second):
    ax,ay,aw,ah=first;bx,by,bw,bh=second
    intersection=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    union=max(0,aw*ah)+max(0,bw*bh)-intersection
    return intersection/union if union>0 else 0


def anatomy_support(hit, samples, minimum=.55):
    """Distinct-frame votes for the same anatomy class and overlapping box.

    These are correlated model detections, not a calibrated probability or a
    tracker. The dedicated detector supplies its own evidence instead of letting
    an uncertain general scene model veto every otherwise clear region.
    """
    if hit['score'] < minimum:
        return []
    return sorted({t for t,hits in samples if any(other['class']==hit['class'] and
        other['score']>=minimum and box_iou(hit['box'],other['box'])>=.3 for other in hits)})


def refine_anatomy(model, proposal, burst, center_index, cache=None):
    """Recheck one weak proposal with context, never promote its weak score alone.

    Use one fixed crop across the burst; remap every detection before matching.
    Center evidence and three distinct matching frames must pass the same .55
    threshold as full-picture detections. This is refinement, not tracking.
    An optional cache shares crop detections between overlapping bursts.
    """
    if not .25 <= proposal['score'] < .55:
        return None
    height,width=burst[center_index][1].shape[:2]
    x,y,w,h=proposal['box']
    size=min(width,height,math.ceil(max(width*.5,3*max(w,h))))
    if size<2:return None
    left=max(0,min(width-size,round(x+w/2-size/2)))
    top=max(0,min(height-size,round(y+h/2-size/2)))
    def detect(pixels):
        if pixels.shape[:2]!=(height,width):
            raise RuntimeError('Anatomy burst dimensions changed')
        key=(id(pixels),left,top,size)
        if cache is None or key not in cache:
            hits=[dict(hit,box=[hit['box'][0]+left,hit['box'][1]+top,*hit['box'][2:]])
                  for hit in model.detect(pixels[top:top+size,left:left+size,::-1].copy())]
            if cache is None:
                return [hit for hit in hits if hit['class']==proposal['class']]
            cache[key]=hits
        return [hit for hit in cache[key] if hit['class']==proposal['class']]
    center_hits=detect(burst[center_index][1])
    matches=[hit for hit in center_hits if hit['score']>=.55 and box_iou(hit['box'],proposal['box'])>=.3]
    if not matches:return None
    anchor=max(matches,key=lambda hit:hit['score'])
    samples=[(t,center_hits if i==center_index else detect(pixels)) for i,(t,pixels) in enumerate(burst)]
    votes=anatomy_support(anchor,samples)
    return (anchor,votes) if len(votes)>=3 else None


def anatomy_proposals(samples, index):
    """At most two weak region proposals from this sample or within one second.

    Neighboring proposals only choose where to look. They never copy an accepted
    filter into another frame; refine_anatomy must verify fresh center evidence.
    """
    center=samples[index][0]
    candidates=[(abs(t-center),-hit['score'],t,hit) for t,hits in samples if abs(t-center)<=1.01
                for hit in hits if .25<=hit['score']<.55]
    candidates.sort(key=lambda item:item[:3])
    selected=[]
    for _,_,t,hit in candidates:
        if any(hit['class']==other['class'] and box_iou(hit['box'],other['box'])>=.3 for _,other in selected):
            continue
        selected.append((t,hit))
        if len(selected)==2:break
    return selected


def _decode_native(path: Path, start: float, end: float):
    """Decode one native timestamp tile, downscaled to at most 640 pixels wide."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        rate = float(stream.average_rate or stream.guessed_rate or 24)
        try:
            container.seek(max(0, int(start / float(stream.time_base))), stream=stream, backward=True)
        except av.error.FFmpegError:
            pass
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-5 < start:
                continue
            if frame.time >= end:
                break
            width = min(640, frame.width)
            height = max(2, int(frame.height * width / frame.width))
            frames.append((float(frame.time), frame.reformat(width=width, height=height, format="rgb24").to_ndarray()))
    return rate, frames


def decoded_range(path, start, end, inclusive=False):
    """Share timestamped native RGB frames across detector rates and overlapping seeks."""
    frames, rate = {}, 24
    source = source_key(path)
    last = math.floor(end / 2) if inclusive else math.ceil(end / 2) - 1
    for tile in range(max(0, math.floor(start / 2)), last + 1):
        rate, decoded = _frame_tiles.get((source, tile), lambda tile=tile:_decode_native(path, tile*2, (tile+1)*2))
        for t, pixels in decoded:
            if t + 1e-5 >= start and (t <= end if inclusive else t < end):
                frames[t] = pixels
    return rate, sorted(frames.items())


@_cached_decode
def native_frames(path: Path, start: float, end: float, target_fps: float):
    """Global-frame sampling from shared native tiles; never interpolate frames."""
    rate, decoded = decoded_range(path, start, end)
    step = max(1, round(rate / target_fps))
    frames = [(t,p) for t,p in decoded if not round(t*rate) % step]
    if not frames:
        raise RuntimeError(f'No decoded frames in interval {start:.2f}–{end:.2f}')
    return frames


def shot_ids(frames, histogram_change=.2, layout_match=.5, flat_histogram_change=.55):
    """Shot number per sampled frame. Temporal evidence never crosses a hard cut.

    A cut needs both a color/brightness change and a changed picture layout.
    On a dark Westworld development scene, flashlight flicker within a shot kept
    layout correlation at .88-.95 while real cuts measured -.27 to .33. Nearly
    flat frames have no reliable layout and use a stricter histogram test.
    Gradual dissolves stay one shot.
    """
    import cv2
    ids, shot, previous = [], 0, None
    for _, pixels in frames:
        small = cv2.resize(pixels, (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        # Brightness matters too: a dark-to-bright cut can keep the same hues.
        histogram = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram)
        layout = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (32, 18),
                                             interpolation=cv2.INTER_AREA).astype(np.float32), (3, 3), 0)
        contrast = float(layout.std())
        layout = (layout - layout.mean()) / (np.linalg.norm(layout - layout.mean()) + 1e-6)
        if previous is not None:
            change = cv2.compareHist(previous[0], histogram, cv2.HISTCMP_BHATTACHARYYA)
            if min(contrast, previous[2]) < 2:
                cut = change > flat_histogram_change
            else:
                cut = change > histogram_change and float((layout * previous[1]).sum()) < layout_match
            shot += cut
        ids.append(shot)
        previous = (histogram, layout, contrast)
    return ids


def shot_neighborhood(shots, index, size=5):
    """The `size` nearest samples in the same shot, centred on `index` where possible."""
    shot = [j for j in range(len(shots)) if shots[j] == shots[index]]
    position = shot.index(index)
    low = max(0, min(position - size // 2, len(shot) - size))
    return shot[low:low + size]


def median_flow(previous, current, box, minimum_points=6, max_error=2.0):
    """Move one box between two grayscale frames with forward-backward checked optical flow.

    Points inside the box are tracked forward and back; only points that return
    within `max_error` pixels vote. The box moves by their median displacement
    and scales by their median spread ratio (limited to ±20%). Returns None when
    too few points survive or the box leaves the picture, so tracking stops
    instead of drifting.
    """
    import cv2
    height, width = previous.shape[:2]
    x, y, w, h = box
    x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(width, int(x + w)), min(height, int(y + h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    mask = np.zeros_like(previous)
    mask[y0:y1, x0:x1] = 255
    points = cv2.goodFeaturesToTrack(previous, 60, .01, 3, mask=mask)
    if points is None or len(points) < minimum_points:
        grid = np.stack(np.meshgrid(np.linspace(x0, x1 - 1, 8), np.linspace(y0, y1 - 1, 8)), -1).reshape(-1, 1, 2)
        points = grid.astype(np.float32)
    points = points.astype(np.float32)
    settings = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **settings)
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **settings)
    error = np.linalg.norm((points - backward).reshape(-1, 2), axis=1)
    good = (status.ravel() == 1) & (back_status.ravel() == 1) & (error < max_error)
    if good.sum() < minimum_points:
        return None
    before, after = points.reshape(-1, 2)[good], forward.reshape(-1, 2)[good]
    dx, dy = np.median(after - before, axis=0)
    spread_before = np.linalg.norm(before - before.mean(0), axis=1)
    spread_after = np.linalg.norm(after - after.mean(0), axis=1)
    usable = spread_before > 1
    scale = float(np.clip(np.median(spread_after[usable] / spread_before[usable]), .8, 1.2)) if usable.sum() >= 3 else 1.0
    cx, cy = x + w / 2 + dx, y + h / 2 + dy
    w, h = w * scale, h * scale
    moved = [cx - w / 2, cy - h / 2, w, h]
    inside = (min(width, moved[0] + w) - max(0, moved[0])) * (min(height, moved[1] + h) - max(0, moved[1]))
    if inside < .5 * w * h:
        return None
    return [round(v, 1) for v in moved]


def track_anatomy(frames, shots, anchors, owned, spacing, bridge=1.0, edge=.5, rejected=()):
    """Extend accepted anatomy regions by tracking them within their shot.

    anchors are accepted detections in found-tuple form. From every anchor whose
    region is not detected in the next sample, the region is tracked forward.
    If it reaches another accepted detection of the same class within `bridge`
    seconds, the whole gap is filled. Otherwise only the first `edge` seconds
    are kept, and the same edge is tracked backward from region starts.
    Tracking never starts a region; it only extends accepted evidence.
    """
    import cv2
    gray = {}

    def frame(i):
        if i not in gray:
            gray[i] = cv2.cvtColor(frames[i][1], cv2.COLOR_RGB2GRAY)
        return gray[i]

    by_index = {}
    for anchor in anchors:
        by_index.setdefault(anchor[0], []).append(anchor)
    denied = {}
    for item in rejected:
        denied.setdefault(item[0], []).append(item[2])

    def blocked(i, cls, box):
        return any(hit['class']==cls and box_iou(hit['box'],box)>=.1 for hit in denied.get(i,[]))

    def detected(i, cls, box):
        return any(a[2]['class'] == cls and box_iou(a[2]['box'], box) >= .1 for a in by_index.get(i, []))

    added = {}
    for index, shot, hit, votes, _, _, _ in anchors:
        for direction, limit in ((1, bridge), (-1, edge)):
            if detected(index + direction, hit['class'], hit['box']):
                continue
            path, box, i = [], hit['box'], index
            while True:
                j = i + direction
                if not 0 <= j < len(frames) or shots[j] != shot or abs(frames[j][0] - frames[index][0]) > limit + 1e-6:
                    break
                box = median_flow(frame(i), frame(j), box)
                if box is None:
                    break
                if blocked(j,hit['class'],box):
                    break
                if detected(j, hit['class'], box):
                    path.append(None)  # Reached another accepted detection: keep the whole bridge.
                    break
                path.append((j, box))
                i = j
            if path and path[-1] is None:
                kept = path[:-1]
            else:
                kept = [step for step in path if abs(frames[step[0]][0] - frames[index][0]) <= edge + 1e-6]
            for j, box in kept:
                key = (j, hit['class'], tuple(round(v) for v in box))
                if owned(j) and key not in added:
                    tracked = dict(**{'class': hit['class']}, score=float(hit['score']), box=[int(round(v)) for v in box])
                    added[key] = (j, shot, tracked, votes, float(hit['score']), frames[index][0], 'tracked')
    return list(added.values())


def anatomy_segments(frames, found, spacing, longest=.5, rejected=(), bounds=None):
    """Chain per-frame anatomy detections into short segments with one conservative region.

    found holds (frame index, shot, hit, votes, proposal score, proposal time,
    refinement). Links need the same class, shot and acceptance, at most one
    skipped sample and overlapping boxes. The region is the union of every
    member box with its margin, so moving anatomy stays covered between
    samples. Segments end after `longest` seconds. This is not a tracker.
    """
    height, width = frames[0][1].shape[:2]
    segments = []
    def blocked_gap(segment,index,hit):
        return any(segment['last']<item[0]<index and item[1]==segment['shot']
                   and item[2]['class']==hit['class']
                   and max(box_iou(item[2]['box'],segment['box']),box_iou(item[2]['box'],hit['box']))>=.1
                   for item in rejected)
    for index, shot, hit, votes, proposal_score, proposal_time, refinement in sorted(found, key=lambda item: item[0]):
        accepted = len(votes) >= 3
        t = frames[index][0]
        match = next((s for s in reversed(segments) if s['class'] == hit['class'] and s['accepted'] == accepted
                      and s['shot'] == shot and 0 < index - s['last'] <= 2 and t - s['times'][0] < longest
                      and not blocked_gap(s,index,hit)
                      and box_iou(s['box'], hit['box']) >= .1), None)
        if match is None:
            match = dict(**{'class': hit['class']}, accepted=accepted, shot=shot, times=[], boxes=[], scores=[],
                         votes=[], proposals=[], refinements=set())
            segments.append(match)
        match.update(last=index, box=hit['box'])
        match['times'].append(t); match['boxes'].append(hit['box']); match['scores'].append(float(hit['score']))
        match['votes'].append(len(votes)); match['proposals'].append((proposal_score, proposal_time))
        match['refinements'].add(refinement)
    events = []
    for s in segments:
        corners = []
        for x, y, w, h in s['boxes']:
            margin_x, margin_y = max(w*.2, width*.015), max(h*.2, height*.015)
            corners.append((max(0, x-margin_x)/width, max(0, y-margin_y)/height,
                            min(width, x+w+margin_x)/width, min(height, y+h+margin_y)/height))
        x1, y1 = min(c[0] for c in corners), min(c[1] for c in corners)
        x2, y2 = max(c[2] for c in corners), max(c[3] for c in corners)
        if x2 <= x1 or y2 <= y1:
            continue
        support = min(s['votes']) if s['accepted'] else max(s['votes'])
        refinement = ' + '.join(sorted(s['refinements']))
        event=dict(start=s['times'][0] - spacing/2, end=s['times'][-1] + spacing/2,
                           category="nudity", confidence=max(s['scores']), sample_time=s['times'][0],
                           region=[x1, y1, x2-x1, y2-y1], review_required=not s['accepted'],
                           support_frames=support, samples=len(s['times']),
                           proposal_confidence=min(p for p, _ in s['proposals']), proposal_time=s['proposals'][0][1],
                           refinement=refinement, source="NudeNet",
                           note=f"{s['class']} · {refinement} · {len(s['times'])} samples, "
                                f"{'each' if s['accepted'] else 'best'} supported on {support}/5 nearby frames")
        if s['accepted'] and rejected:
            related=[frames[item[0]][0] for item in rejected if item[1]==s['shot'] and item[2]['class']==s['class']
                     and any(box_iou(item[2]['box'],box)>=.1 for box in s['boxes'])]
            before=[t+spacing/2 for t in related if t<s['times'][0]]
            after=[t-spacing/2 for t in related if t>s['times'][-1]]
            if before or after:
                low,high=bounds or (0.,frames[-1][0]+spacing/2)
                event['time_bounds']=[max([low]+before),min([high]+after)]
        events.append(event)
    return events


def keyword_events(words, phrases, allowed):
    """Whole-token phrases; an allowed phrase exempts its complete word span."""
    expanded = []
    for word in words:
        expanded.extend((token, *word[1:]) for token in tokens(word[0]))
    exempt = set()
    def spans(phrase):
        needle = tokens(phrase)
        for i in range(len(expanded) - len(needle) + 1):
            span = expanded[i:i + len(needle)]
            if needle and [w[0] for w in span] == needle and not any(b[1] - a[2] > 1 for a, b in zip(span, span[1:])):
                yield i, span
    for phrase in allowed:
        for i, span in spans(phrase):
            exempt.update(range(i, i + len(span)))
    events = []
    for phrase in phrases:
        for i, span in spans(phrase):
            if any(j in exempt for j in range(i, i + len(span))) or span[-1][2] <= span[0][1]:
                continue
            events.append(dict(start=span[0][1], end=span[-1][2], category="language",
                               confidence=min(w[3] for w in span), source="Whisper keyword", note=phrase))
    return events


def _loading(method):
    """Serialize model loading; returns quickly once a model is resident."""
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    wrapper.__name__, wrapper.__doc__ = method.__name__, method.__doc__
    return wrapper


class Models:
    """Resident models shared by concurrent analyses.

    `lock` guards model loading. Chunks run concurrently, with separate limits
    for learned GPU inference and CUDA anatomy calls. `workers` sets how many
    concurrent analyses the scene verifier serves in parallel slots.
    """
    def __init__(self, workers=1, evaluate_unpromoted=False):
        self.root = prepare_environment()
        self.sessions = {}
        self.workers = workers
        # Offline evaluation may run a head set before promotion; the player never does.
        self.evaluate_unpromoted = evaluate_unpromoted
        self.lock = threading.RLock()
        self.dll_handles = []
        self.backends = {}
        self.warnings = []
        self.feature_models = None

    def ort(self, device):
        import onnxruntime as ort
        if os.name == "nt" and not self.dll_handles:
            import site
            for parent in site.getsitepackages():
                for directory in Path(parent).glob("nvidia/*/bin"):
                    self.dll_handles.append(os.add_dll_directory(str(directory)))
                    # cuDNN loads its plugin libraries through LoadLibrary internally.
                    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
        available = ort.get_available_providers()
        gpu = device == "cuda" and "CUDAExecutionProvider" in available
        if device == "cuda" and not gpu:
            raise RuntimeError("CUDA requested but ONNX Runtime CUDA provider is unavailable")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        providers = [("CUDAExecutionProvider", {"device_id": 0, "gpu_mem_limit": 3 * 1024**3,
                                                "cudnn_conv_algo_search": "HEURISTIC"}),
                     "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        return ort, options, providers

    @_loading
    def speech(self, device, name="tiny.en"):
        key = ("speech", device, name)
        if key not in self.sessions:
            from faster_whisper import WhisperModel
            import ctranslate2
            self.ort(device)  # Load package-local CUDA DLLs on Windows.
            gpu = device == "cuda" and ctranslate2.get_cuda_device_count() > 0
            try:
                model_path = speech_model_path(self.root, model=name)
            except Exception as error:
                raise RuntimeError(f"Speech model {name} is not installed. Run model_setup.py --speech-model {name} once, or select Fast speech.") from error
            try:
                model = WhisperModel(model_path, device="cuda" if gpu else "cpu",
                                     compute_type="float16" if gpu else "int8", cpu_threads=4,
                                     download_root=str(self.root / "models"), local_files_only=True)
            except Exception as error:
                if device == "cuda" or not gpu:
                    raise
                self.warnings.append(f"Speech GPU unavailable; using CPU: {type(error).__name__}")
                model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=4,
                                     download_root=str(self.root / "models"), local_files_only=True)
                gpu = False
            self.sessions[key] = model
        self.backends["speech"] = f"{name} / " + ("CUDA float16" if self.sessions[key].model.device == "cuda" else "CPU int8")
        return self.sessions[key]

    @_loading
    def nude(self, device, name="320n"):
        import nudenet
        path = Path(nudenet.__file__).parent / '320n.onnx' if name == '320n' else self.root / 'scene-verifier/640m.onnx'
        key = ("nudity", device, name, file_hash(path))
        if key not in self.sessions:
            import nudenet
            resolved = device
            if device == "auto":
                # 320n measured 4.6 ms on CPU versus 11 ms on CUDA; 640m measured
                # 91 ms on CPU versus 10 ms on CUDA. A cold CUDA cache can make the
                # first inference take ~30 s, so warm_up() runs at engine start.
                import onnxruntime
                cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
                resolved = "cuda" if name == "640m" and cuda else "cpu"
            ort, options, providers = self.ort(resolved)
            # NudeNet 3.4.2 ignores its providers argument. Construct its documented
            # inference fields ourselves so thread counts and providers are explicit.
            model = nudenet.NudeDetector.__new__(nudenet.NudeDetector)
            path = Path(nudenet.__file__).parent / "320n.onnx" if name == "320n" else self.root / 'scene-verifier/640m.onnx'
            if not path.is_file():
                raise RuntimeError('Install the detailed anatomy model with setup_verifier.py, or choose Fast anatomy.')
            model.onnx_session = ort.InferenceSession(str(path),
                                                     sess_options=options, providers=providers)
            model.input_width = model.input_height = 320 if name == '320n' else 640
            model.input_name = model.onnx_session.get_inputs()[0].name
            from functools import partial
            from anatomy_runtime import detect
            infer = partial(detect, model)
            if model.onnx_session.get_providers()[0] == 'CUDAExecutionProvider':
                # Concurrent runs changed weak proposals/rounded boxes in the
                # six-title benchmark. Share loading, serialize this GPU detector.
                detector_lock = threading.Lock()
                def stable_detect(image, include_class_scores=False):
                    with detector_lock:
                        return infer(image,include_class_scores=include_class_scores)
                model.detect = stable_detect
            else:
                model.detect = infer
            if device == "cuda":
                model.onnx_session.disable_fallback()
            self.sessions[key] = model
            for old in list(self.sessions):
                if old[:3] == key[:3] and old != key:
                    del self.sessions[old]
            actual = model.onnx_session.get_providers()
            self.backends["nudity"] = actual[0]
            if device == "cuda" and actual[0] != "CUDAExecutionProvider":
                raise RuntimeError("Nudity model could not activate CUDA")
        return self.sessions[key]

    def warm_up(self, names=('320n', '640m')):
        """Initialize anatomy sessions before the first chunk needs them."""
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        for name in names:
            try:
                with self.lock:
                    self.nude('auto', name).detect(blank)
            except Exception as error:  # An optional model may not be installed.
                self.warnings.append(f'Anatomy {name} warm-up skipped: {type(error).__name__}')

    @_loading
    def verifier(self, name='4B'):
        key=('verifier',name)
        if key not in self.sessions:
            # Only one vision-language model may occupy GPU memory at a time.
            for old in list(self.sessions):
                if old[0]=='verifier':
                    self.sessions.pop(old).close()
            from scene_verifier import SceneVerifier
            self.sessions[key] = SceneVerifier(self.root,model_size=name,slots=self.workers)
        return self.sessions[key]

    def close(self):
        for key,model in self.sessions.items():
            if key[0]=='verifier':
                model.close()

    @_loading
    def clip(self, device, custom):
        key = ("clip", device)
        root = self.root / 'clip'
        stamp = digest({str(p.relative_to(root)):file_hash(p) for p in list(root.rglob('*.onnx')) + list(root.glob('*.json'))})
        if self.sessions.get(('clip_stamp', device)) != stamp:
            self.sessions.pop(key, None)
            for old in list(self.sessions):
                if old[0] == 'prompts':
                    del self.sessions[old]
            self.sessions[('clip_stamp', device)] = stamp
        if key not in self.sessions:
            from tokenizers import Tokenizer
            ort, options, providers = self.ort(device)
            root = self.root / "clip"
            use_gpu = providers[0] != "CPUExecutionProvider" and (root / "onnx/vision_model.onnx").exists()
            vision_file = "vision_model.onnx" if use_gpu else "vision_model_quantized.onnx"
            if use_gpu and (root / "onnx/vision_model_matmul.onnx").exists():
                vision_file = "vision_model_matmul.onnx"
            vision = ort.InferenceSession(str(root / "onnx" / vision_file), sess_options=options,
                                          providers=providers if use_gpu else ["CPUExecutionProvider"])
            text = ort.InferenceSession(str(root / "onnx/text_model_quantized.onnx"), sess_options=options,
                                        providers=["CPUExecutionProvider"])
            tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=77)
            tokenizer.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
            self.sessions[key] = (vision, text, tokenizer)
            if device == "cuda":
                vision.disable_fallback()
            self.backends["scenes"] = vision.get_providers()[0]
            if device == "cuda" and vision.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("Scene CUDA model missing; run model_setup.py --gpu")
        vision, text, tokenizer = self.sessions[key]
        text_key = ("prompts", custom)
        if text_key not in self.sessions:
            prompts = dict(SCENE_PROMPTS)
            if custom.strip():
                prompts["custom"] = [custom.strip()]
            names = [name for name, entries in prompts.items() for _ in entries] + ["neutral"] * len(NEUTRAL_PROMPTS)
            sentences = [sentence for entries in prompts.values() for sentence in entries] + NEUTRAL_PROMPTS
            encoded = tokenizer.encode_batch(sentences)
            inputs = {"input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                      "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64)}
            values = text.run(None, {i.name: inputs[i.name] for i in text.get_inputs()})
            output_index = next((i for i, out in enumerate(text.get_outputs()) if "embeds" in out.name), 0)
            embedding = values[output_index]
            embedding /= np.maximum(np.linalg.norm(embedding, axis=-1, keepdims=True), 1e-9)
            # Keep at most the current text bank; custom changes do not grow GPU memory.
            for old in list(self.sessions):
                if old[0] == "prompts":
                    del self.sessions[old]
            self.sessions[text_key] = (names, embedding)
        return vision, self.sessions[text_key]

    def clip_embeddings(self, frames, device, custom='', fit='crop'):
        """Unit-length CLIP image embeddings. 'crop' matches CLIP's center crop;
        'squash' resizes the whole picture so content near the edges is kept."""
        from PIL import Image
        vision, _ = self.clip(device, custom)
        means = np.asarray([.48145466, .4578275, .40821073], dtype=np.float32)
        stds = np.asarray([.26862954, .26130258, .27577711], dtype=np.float32)
        index = next((i for i, out in enumerate(vision.get_outputs()) if "embeds" in out.name), 0)
        results = []
        for offset in range(0, len(frames), 8):
            batch = []
            for _, frame in frames[offset:offset + 8]:
                image = Image.fromarray(frame)
                if fit == 'squash':
                    image = image.resize((224, 224), Image.Resampling.BICUBIC)
                else:
                    scale = 224 / min(image.size)
                    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
                    x, y = (image.width - 224) // 2, (image.height - 224) // 2
                    image = image.crop((x, y, x + 224, y + 224))
                pixels = np.asarray(image, dtype=np.float32) / 255
                batch.append(((pixels - means) / stds).transpose(2, 0, 1))
            embeddings = vision.run(None, {vision.get_inputs()[0].name: np.stack(batch)})[index]
            self.backends["scenes"] = vision.get_providers()[0]
            results.append(embeddings / np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-9))
        return np.concatenate(results) if results else np.zeros((0, 512), dtype=np.float32)

    @_loading
    def text_embeddings(self, prompts, device='cpu'):
        """Unit-length CLIP text embeddings for arbitrary prompts."""
        self.clip(device, '')
        _, text, tokenizer = self.sessions[("clip", device)]
        encoded = tokenizer.encode_batch(list(prompts))
        inputs = {"input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                  "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64)}
        values = text.run(None, {i.name: inputs[i.name] for i in text.get_inputs()})
        embedding = values[next((i for i, out in enumerate(text.get_outputs()) if "embeds" in out.name), 0)]
        return embedding / np.maximum(np.linalg.norm(embedding, axis=-1, keepdims=True), 1e-9)

    def scene_scores(self, frames, device, custom, embeddings=None):
        _, (names, texts) = self.clip(device, custom)
        results = []
        if embeddings is None:
            embeddings = self.clip_embeddings(frames, device, custom)
        for row in embeddings @ texts.T:
            neutral = max(float(row[i]) for i, name in enumerate(names) if name == "neutral")
            scores = {}
            for name in dict.fromkeys(names):
                if name != "neutral":
                    positive = max(float(row[i]) for i, n in enumerate(names) if n == name)
                    # Heuristic relative similarity, deliberately NOT called probability.
                    scores[name] = float(1 / (1 + math.exp(-20 * (positive - neutral))))
            results.append(scores)
        return results

    def analyze(self, path, start, end, duration, config, has_audio, trace=None):
        """Events owned by [start, end). Optional trace collects first-stage candidates.

        Trace entries record what each detector proposed before gating, so
        candidate recall can be measured separately from final decisions.
        """
        left, right = max(0, start - CONTEXT), min(duration, end + CONTEXT)
        record = trace.append if trace is not None else (lambda entry: None)
        events = []
        if config.speech and has_audio and config.phrases:
            events += self._speech_events(path, left, right, config, record)
        if config.scenes and config.scene_mode == 'learned' and not config.learned_heads:
            raise RuntimeError('Learned scene analysis needs a promoted learned detector set.')
        if config.nudity or config.scenes:
            frames = sample_frames(path, left, right, config.fps) if config.scenes and config.scene_mode != 'learned' else []
            bursts = {}
            if config.scenes and config.scene_mode == 'verified' and frames:
                events += self._verified_scene_events(path, frames, start, end, config, bursts, record)
            if config.nudity:
                events += self._anatomy_events(path, start, end, duration, config, record)
                events = reconcile_nudity(events)
            if config.scenes and config.scene_mode == 'review':
                events += self._clip_events(frames, start, end, config, record)
        if config.learned_heads:
            events += self._learned_events(path, start, end, duration, config, record)
        if path.is_file() and any(e['category'] != 'language' and 'time_bounds' not in e for e in events):
            boundary_frames = native_frames(path, max(0,start-1), min(duration,end+1), config.anatomy_fps)
            ids = shot_ids(boundary_frames)
            cuts = [boundary_frames[i][0] for i in range(1,len(ids)) if ids[i] != ids[i-1]]
            for event in events:
                if event['category'] == 'language' or 'time_bounds' in event:
                    continue
                t=event.get('sample_time',(event['start']+event['end'])/2)
                low=max([0.] + [cut for cut in cuts if cut<=t])
                high=min([duration] + [cut for cut in cuts if cut>t])
                event.update(start=max(low,event['start']),end=min(high,event['end']),time_bounds=[low,high])
        clean = []
        for event in events:
            event["start"], event["end"] = max(0, event["start"]), min(duration, event["end"])
            owner_time = event.get('sample_time', (event['start']+event['end'])/2)
            if event["end"] > event["start"] and start <= owner_time < end:
                event.update(enabled=True, severity=2)
                clean.append(event)
        return merge_scene_events(clean)

    def _speech_events(self, path, left, right, config, record):
        samples = audio_samples(path, left, right)
        model = self.speech(config.device, config.speech_model)
        segments, _ = model.transcribe(samples, word_timestamps=True, vad_filter=True,
                                        beam_size=5 if config.speech_model == "small.en" else 1,
                                        condition_on_previous_text=False, language="en")
        words = [(w.word, w.start + left, w.end + left, w.probability)
                 for segment in segments for w in segment.words or []]
        events = keyword_events(words, config.phrases, config.allow_phrases)
        for event in events:
            record(dict(stage='speech_keyword', category='language', start=event['start'],
                        end=event['end'], score=event['confidence'], detail=event['note']))
        return events

    def _verified_scene_events(self, path, frames, start, end, config, bursts, record):
        # At most 20 sequence frames per 10-second contextual window.
        # Anatomy still receives every configured visual sample.
        indices = np.linspace(0,len(frames)-1,min(20,len(frames)),dtype=int)
        sequence = [frames[i] for i in indices]
        verifier=self.verifier(config.scene_model)
        result = verifier.verify(sequence,config.custom_visual)
        self.backends['scenes'] = f'Qwen3-VL {config.scene_model} / CUDA sequence verification'
        spacing = float(np.median(np.diff([t for t,_ in sequence]))) if len(sequence)>1 else 1/config.fps
        events=[]
        candidates=set()
        for tag in result['tags']:
            if tag['category'] == 'nudity' and not config.nudity:
                continue
            if tag['category'] == 'custom' and not config.custom_visual.strip():
                continue
            if tag['state']=='absent':
                continue
            for i in tag['frames']:
                t=sequence[i][0]
                if start<=t<end:
                    record(dict(stage='scene_sequence',category=tag['category'],start=t-spacing/2,end=t+spacing/2,
                                sample_time=t,state=tag['state'],detail=tag['reason']))
            if tag['category']=='nudity':
                # Only the dedicated anatomy pass can supply automatic
                # regions. Retain these broad proposals for review without
                # paying for a VLM burst that cannot accept a region.
                for i in tag['frames']:
                    t=sequence[i][0]
                    if start<=t<end:
                        events.append(dict(start=t-spacing/2,end=t+spacing/2,sample_time=t,
                            category='nudity',confidence=1,source='Qwen sequence proposal',
                            evidence='sequence_proposal',review_required=True,
                            note='Unlocalized sequence suggestion: '+tag['reason'],score_type='proposal'))
                continue
            for i in tag['frames']:
                # Check neighboring action samples too: a broad proposal
                # can locate the scene but miss the precise contact frame.
                neighbors=range(max(0,i-1),min(len(sequence),i+2)) if tag['category']=='violence' else [i]
                candidates.update((tag['category'],j) for j in neighbors if start<=sequence[j][0]<end)
        for category,i in sorted(candidates):
            t=sequence[i][0]
            if t not in bursts:
                bursts[t]=sample_burst(path,t)
            context,center_index=bursts[t]
            evidence=verifier.check_frame(context,category,config.custom_visual,center_index=center_index)
            record(dict(stage='scene_evidence',category=category,start=t-spacing/2,end=t+spacing/2,
                        sample_time=t,state=evidence['state'],detail=evidence['observation']))
            if evidence['state']=='absent':
                continue
            events.append(dict(start=t-spacing/2,end=t+spacing/2,sample_time=t,
                category=category,confidence=1,source='Qwen verified frame',evidence=evidence['label'],
                review_required=evidence['state']=='uncertain' or config.scene_filtering=='review',
                note=evidence['observation'],score_type='decision'))
        return events

    def _anatomy_events(self, path, start, end, duration, config, record):
        """Dense anatomy: every owned native sample gets its own five-frame support check.

        Support comes from the nearest distinct samples in the same shot, so the
        3-of-5 rule now applies about every 83 ms instead of every 0.5 s. Weak
        proposals within one second choose where to recheck a contextual crop;
        they never copy a decision. Accepted and reviewable hits are chained
        into short segments.
        """
        model = self.nude(config.device,config.anatomy_model)
        gate=self.roi_gate(config.anatomy_roi,config.anatomy_model) if config.anatomy_roi else None
        detect=model.detect
        if gate and (gate.head.get('feature_kind')=='detector-scores' or 'detector_head' in gate.head):
            from functools import partial
            detect=partial(detect,include_class_scores=True)
        frames = native_frames(path, max(0, start - 1), min(duration, end + 1), config.anatomy_fps)
        shots = shot_ids(frames)
        hits = [[hit for hit in detect(pixels[:,:,::-1].copy()) if hit['class'] in NUDITY_CLASSES]
                for _, pixels in frames]
        self.backends["nudity"] = model.onnx_session.get_providers()[0]
        spacing = float(np.median(np.diff([t for t, _ in frames]))) if len(frames) > 1 else 1 / config.anatomy_fps
        crops, found = {}, []
        for i, (t, _) in enumerate(frames):
            if not start <= t < end:
                continue
            for hit in hits[i]:
                record(dict(stage='anatomy_raw',category='nudity',start=t-spacing/2,end=t+spacing/2,
                            sample_time=t,score=float(hit['score']),detail=hit['class']))
            shot = [j for j in range(len(frames)) if shots[j] == shots[i]]
            neighborhood = shot_neighborhood(shots, i)
            burst = [frames[j] for j in neighborhood]
            local = [(frames[j][0], hits[j]) for j in neighborhood]
            nearby = [j for j in shot if abs(frames[j][0] - t) <= 1.01]
            refined = []
            for proposal_time, proposal in anatomy_proposals([(frames[j][0], hits[j]) for j in nearby], nearby.index(i)):
                result = refine_anatomy(model, proposal, burst, neighborhood.index(i), crops)
                if result is not None:
                    refined.append((*result, float(proposal['score']), proposal_time))
            for hit in hits[i]:
                # Keep weak proposals reviewable unless refinement supplies an accepted replacement.
                if hit['score'] < .55 and any(hit['class'] == anchor['class'] and box_iou(hit['box'], anchor['box']) >= .3
                                              for anchor, *_ in refined):
                    continue
                found.append((i, shots[i], hit, anatomy_support(hit, local), float(hit['score']), t, 'full picture'))
            found += [(i, shots[i], anchor, votes, score, proposal_time, 'contextual crop')
                      for anchor, votes, score, proposal_time in refined]
        rejected=[]
        if config.anatomy_tracking and found:
            # Context samples can anchor a bridge into this chunk; only owned samples are emitted.
            context=[]
            for i, (t, _) in enumerate(frames):
                if not start <= t < end:
                    local = [(frames[j][0], hits[j]) for j in shot_neighborhood(shots, i)]
                    context += [(i, shots[i], hit, votes, float(hit['score']), t, 'full picture') for hit in hits[i]
                                if len(votes := anatomy_support(hit, local)) >= 3]
            if gate:
                # One decision batch includes owned samples and boundary context,
                # so temporal votes cannot disagree between detection and tracking.
                combined=gate.refine(frames,found+context)
                found,context=combined[:len(found)],combined[len(found):]
                rejected=[item for item in combined if item[2].get('roi_rejected')]
            anchors=[item for item in found+context if len(item[3])>=3]
            if anchors:
                found += track_anatomy(frames, shots, anchors, lambda j: start <= frames[j][0] < end, spacing,rejected=rejected)
        if gate and not config.anatomy_tracking:
            found = gate.refine(frames, found)
            rejected=[item for item in found if item[2].get('roi_rejected')]
        events = anatomy_segments(frames, found, spacing,rejected=rejected,bounds=(0,duration))
        # Context edges are not cuts. Only actual changes of shot bound padding.
        cuts = [frames[i][0] for i in range(1, len(frames)) if shots[i] != shots[i-1]]
        for event in events:
            t = event['sample_time']
            low = max([0.0] + [cut for cut in cuts if cut <= t])
            high = min([duration] + [cut for cut in cuts if cut > t])
            if 'time_bounds' in event:
                low=max(low,event['time_bounds'][0]);high=min(high,event['time_bounds'][1])
            event.update(start=max(low, event['start']), end=min(high, event['end']), time_bounds=[low, high])
        return events

    @_loading
    def roi_gate(self, name, anatomy_model=''):
        from roi_gate import AnatomyGate
        path = self.root/'roi-models'/f'{name}.json'
        key = ('roi',name,file_hash(path),file_hash(self.root/'clip/onnx/vision_model_quantized.onnx'),file_hash(self.root/'scene-verifier/640m.onnx'))
        if key not in self.sessions:
            self.sessions[key] = AnatomyGate(self,path)
            for old in list(self.sessions):
                if old[:2] == key[:2] and old != key:
                    del self.sessions[old]
        if anatomy_model and self.sessions[key].head.get('anatomy_model',anatomy_model)!=anatomy_model:
            raise ValueError('This region check requires '+self.sessions[key].head['anatomy_model']+' anatomy')
        return self.sessions[key]

    @_loading
    def learned_features(self):
        """ONNX feature models, created once; concurrent first use is serialized."""
        folder = self.root / 'learned-runtime'
        files = [folder/'manifest.json'] + list(folder.glob('*.onnx')) + list((folder/'ast-extractor').glob('*.json'))
        stamp = digest({p.name:file_hash(p) for p in files})
        if self.feature_models is None or getattr(self, '_feature_stamp', None) != stamp:
            import learned_runtime as lr
            self.feature_models = lr.FeatureModels(self, self.root)
            self._feature_stamp = stamp
        return self.feature_models

    @_loading
    def heads(self, name):
        """A trained head set from the runtime folder, reloaded when its file changes."""
        path = self.root / 'heads' / f'{name}.json'
        if not path.is_file():
            raise RuntimeError(f'Learned detector set {name!r} is not installed.')
        from provenance import file_hash
        key = ('heads', name, file_hash(path))
        if key not in self.sessions:
            heads = json.loads(path.read_text(encoding='utf-8'))
            if not heads.get('promoted') and not self.evaluate_unpromoted:
                raise RuntimeError(f'Learned detector set {name!r} has not been promoted after a labeled comparison.')
            for old in [k for k in self.sessions if k[0] == 'heads' and k[1] == name]:
                del self.sessions[old]
            self.sessions[key] = heads
        return self.sessions[key]

    def _learned_events(self, path, start, end, duration, config, record):
        """Trained heads over CLIP, V-JEPA 2 and audio features (see learned_runtime).

        Head decisions above their validated precision threshold are events;
        candidates below it get the scene model's five-frame evidence check when
        sequence verification is enabled. Decisions follow the scene review
        setting, so 'Review before filtering' applies to them too.
        """
        import learned_runtime as lr
        heads = self.heads(config.learned_heads)
        if heads.get('attention_model'):
            from attention_probe import RuntimeProbe
            key=('attention',heads['attention_model']['sha256'])
            with self.lock:
                if key not in self.sessions:self.sessions[key]=RuntimeProbe(self,heads['attention_model'])
                probe=self.sessions[key]
            times,matrix=probe.matrix(path,start,end,duration)
        else:
            features_models = self.learned_features()
            from player_inputs import chunk_inputs
            times, matrix = chunk_inputs(features_models,path,start,end,duration,heads['kinds'],heads.get('temporal_offsets',()))
        self.backends['learned'] = f"{config.learned_heads} / ONNX CUDA"
        entries, events = lr.head_outputs(heads, times, matrix)
        for entry in entries:
            record(entry)
        if config.scenes and config.scene_mode in ('verified', 'learned') and config.learned_verification in ('selective','confirm'):
            verifier = self.verifier(config.scene_model)

            bursts = {}

            def check(category, middle):
                # Spans of several categories often share a moment; decode its burst once.
                if middle not in bursts:
                    bursts[middle] = sample_burst(path, middle)
                frames, center = bursts[middle]
                return verifier.check_frame(frames, category, config.custom_visual, center_index=center)

            if config.learned_verification=='confirm':
                from learned_confirmation import confirm_outputs
                checked,events=confirm_outputs(entries,events,check,start,end)
                verified=[]
            else:
                spans = lr.uncertain_spans(dict(trace=entries, events=events), stages=('classifier',))
                checked, verified = lr.verified_events(spans, check, start, end)
            for entry in checked:
                record(entry)
            events += verified
        certification = qualified(heads, config.learned_verification)
        if certification:
            certificate = heads['promoted']
            certification = (certificate.get('qualification_identity') == qualification_identity(pipeline_identity(self.root, config, self.workers)))
        for event in events:
            event['review_required'] = (event['review_required'] or config.scene_filtering == 'review'
                                        or (not certification and not self.evaluate_unpromoted)
                                        or (event['category'] == 'nudity' and not event.get('region')))
        return events

    def _clip_events(self, frames, start, end, config, record):
        scores = self.scene_scores(frames, config.device, config.custom_visual)
        events=[]
        for (t, _), row in zip(frames, scores):
            for category, score in row.items():
                if start<=t<end:
                    record(dict(stage='clip',category=category,start=t-1/config.fps,end=t+1/config.fps,
                                sample_time=t,score=score))
                # Retain raw scores so sensitivity edits never rerun inference.
                events.append(dict(start=t - 1 / config.fps, end=t + 1 / config.fps,
                                   category=category, confidence=score, source="CLIP experimental", sample_time=t,
                                   review_required=True,
                                   note="Image-prompt similarity; not a calibrated probability"))
        return events


def reconcile_nudity(events):
    """Prefer localized anatomy evidence over broad scene-model nudity claims."""
    # When anatomy supplies a box for an accepted sample, use the
    # localized evidence rather than an additional whole-frame mask.
    localized = [e for e in events if e['source']=='NudeNet' and not e.get('review_required') and e['confidence']>=.55]
    events = [e for e in events if not (e['source'] in ('Qwen verified frame','Qwen sequence proposal') and e['category']=='nudity'
        and any(e['start']<n['end'] and n['start']<e['end'] for n in localized))]
    # A language-model anatomy claim without a usable anatomy box
    # must never automatically conceal the whole picture.
    for event in events:
        if event['source'] in ('Qwen verified frame','Qwen sequence proposal') and event['category']=='nudity':
            event['review_required']=True
    return events

def merge_scene_events(events):
    """One review action per contiguous scene interval, without bridging clean gaps."""
    grouped={}
    other=[]
    for event in events:
        if event['source']=='Qwen verified frame':
            key=(event['category'],event['review_required'],event.get('evidence'),tuple(event.get('time_bounds',[])))
            grouped.setdefault(key,[]).append(event)
        else:
            other.append(event)
    for group in grouped.values():
        merged=[]
        for event in sorted(group,key=lambda e:e['start']):
            if merged and event['start']<=merged[-1]['end']+.01:
                merged[-1]['end']=max(merged[-1]['end'],event['end'])
            else:
                merged.append(dict(event))
        other.extend(merged)
    for event in other:
        event['id']=hashlib.sha1(json.dumps(event,sort_keys=True).encode()).hexdigest()[:16]
    return sorted(other,key=lambda e:(e['start'],e['end']))


@dataclass
class MediaJob:
    id: str
    title: str
    path: Path
    duration: float
    has_audio: bool
    config: ScanConfig = field(default_factory=ScanConfig)
    position: float = 0
    active: bool = False
    complete_scan: bool = False
    generation: int = 0
    status: str = "ready"
    error: str | None = None
    chunks: dict = field(default_factory=dict)
    timings: list = field(default_factory=list)
    analysis_identity: str | None = None
    wall_seconds: float = 0
    wall_end: float = 0  # monotonic time; never restored across processes
    manual: list = field(default_factory=list)
    ignored: set = field(default_factory=set)
    confirmed: set = field(default_factory=set)
    last_seen: float = field(default_factory=time.monotonic)
    updated: int = 0
    live: bool = False
    segments: dict = field(default_factory=dict)
    capture_running: bool = False
    retained_start: float = 0
    running: set = field(default_factory=set)  # chunk indices being analyzed now

    def coverage(self):
        intervals = []
        for index in sorted(self.chunks):
            left, right = index * CHUNK, min(self.duration, (index + 1) * CHUNK)
            if intervals and left <= intervals[-1][1] + .001:
                intervals[-1][1] = right
            else:
                intervals.append([left, right])
        return intervals

    def next_chunk(self):
        if not self.active or self.error:
            return None
        first = max(0, int(max(self.position, self.retained_start) // CHUNK))
        last = min(math.ceil(self.duration / CHUNK), math.ceil((self.position + self.config.lookahead) / CHUNK))
        candidates = list(range(first, last))
        if self.complete_scan:
            candidates += list(range(math.ceil(self.duration / CHUNK)))
        return next((i for i in candidates if i not in self.chunks and i not in self.running and
                     (not self.live or i in self.segments and (i + 1 in self.segments or not self.capture_running))), None)

    def snapshot(self, models):
        unique = {e["id"]: e for group in self.chunks.values() for e in group}
        unique.update({e["id"]: e for e in self.manual})
        events = [dict(e, enabled=e.get("enabled", True) and e["id"] not in self.ignored,
                       confirmed=e['id'] in self.confirmed) for e in unique.values()]
        media_seconds = sum(min(CHUNK, self.duration - i * CHUNK) for i in self.chunks)
        processing = sum(self.timings)
        return dict(id=self.id, title=self.title, duration=self.duration, has_audio=self.has_audio,
                    live=self.live, capture_running=self.capture_running, retained_start=self.retained_start,
                    config=self.config.model_dump(), generation=self.generation, status=self.status, error=self.error,
                    coverage=self.coverage(), events=sorted(events, key=lambda e: e["start"]), updated=self.updated,
                    analyzed_seconds=media_seconds, processing_seconds=processing,
                    analysis_wall_seconds=self.wall_seconds,
                    speed=media_seconds / self.wall_seconds if self.wall_seconds else None,
                    service_speed=media_seconds / processing if processing else None,
                    p95_chunk_seconds=float(np.percentile(self.timings, 95)) if self.timings else None,
                    backends=dict(models.backends), warnings=list(models.warnings))


class Engine:
    """Schedules chunks for every open media session.

    `workers` chunks are analyzed at once (ADAPTIVE_CENSOR_WORKERS, default 2);
    the most recently controlled media is served first. Two workers measured
    0.84 s per media second against 1.25 s for one with sequence verification.
    """
    def __init__(self, workers=None):
        workers = int(os.environ.get("ADAPTIVE_CENSOR_WORKERS", "2")) if workers is None else workers
        self.models = Models(workers=max(1, workers))
        self.worker_count = max(1, workers)
        self.jobs = {}
        self.condition = threading.Condition()
        self.closed = False
        self.restore()
        self.threads = [threading.Thread(target=self.loop, name=f"analysis-worker-{n}", daemon=True)
                        for n in range(max(1, workers))]
        for thread in self.threads:
            thread.start()
        if callable(getattr(self.models, "warm_up", None)) and getattr(self.models, "root", None) is not None:
            threading.Thread(target=self.models.warm_up, name="model-warm-up", daemon=True).start()

    def restore(self):
        root = getattr(self.models, "root", None)
        if root is None:
            return
        for manifest in (root / "sessions-v2").glob("*/state.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                source = manifest.parent / "source.mp4"
                info = media_info(source)
                job = MediaJob(str(data["id"]), str(data["title"]), source, info["duration"], info["has_audio"],
                                config=ScanConfig(**data["config"]), generation=int(data["generation"]))
                if not re_full_id(job.id):
                    continue
                job.manual = data.get("manual", [])
                job.ignored = set(data.get("ignored", []))
                job.confirmed = set(data.get('confirmed', []))
                job.timings = data.get("timings", [])
                job.wall_seconds = data.get('wall_seconds', 0)
                job.analysis_identity = data.get('analysis_identity')
                for index in data.get("completed", []):
                    index = int(index)
                    if index < 0 or index * CHUNK >= job.duration:
                        continue
                    chunk = source.parent / "chunks" / f"{job.generation}-{index}.json"
                    if chunk.is_file():
                        job.chunks[index] = json.loads(chunk.read_text(encoding="utf-8"))
                job.status = "complete" if len(job.chunks) == math.ceil(job.duration / CHUNK) else "ready"
                if (data.get('analysis_revision') != ANALYSIS_REVISION or data.get('verifier_revision') != VERIFIER_REVISION
                        or job.analysis_identity != self.identity(job)):
                    job.chunks.clear()
                    job.timings.clear()
                    job.wall_seconds = job.wall_end = 0
                    job.analysis_identity = self.identity(job)
                    job.ignored.intersection_update(e['id'] for e in job.manual)
                    job.confirmed.intersection_update(e['id'] for e in job.manual)
                    job.generation += 1
                    job.status = 'ready'
                    self.persist(job)
                self.jobs[job.id] = job
            except (OSError, ValueError, TypeError, KeyError, av.error.FFmpegError):
                self.models.warnings.append(f"Could not restore a saved media session in {manifest.parent.name}")

    def persist(self, job, index=None):
        if job.live or not job.path.is_file():
            return
        directory = job.path.parent
        if index is not None:
            chunks = directory / "chunks"
            chunks.mkdir(exist_ok=True)
            (chunks / f"{job.generation}-{index}.json").write_text(json.dumps(job.chunks[index]), encoding="utf-8")
        if job.analysis_identity is None:
            job.analysis_identity = self.identity(job)
        data = dict(version=3, analysis_identity=job.analysis_identity, wall_seconds=job.wall_seconds,
                    analysis_revision=ANALYSIS_REVISION, verifier_revision=VERIFIER_REVISION, id=job.id, title=job.title, config=job.config.model_dump(), generation=job.generation,
                    manual=job.manual, ignored=list(job.ignored), confirmed=list(job.confirmed), timings=job.timings, completed=list(job.chunks))
        temporary = directory / "state.tmp"
        temporary.write_text(json.dumps(data), encoding="utf-8")
        temporary.replace(directory / "state.json")

    def add(self, path, title):
        info = media_info(path)
        job = MediaJob(uuid.uuid4().hex, title, path, info["duration"], info["has_audio"])
        with self.condition:
            self.jobs[job.id] = job
            self.persist(job)
        return job

    def configure(self, job, config):
        with self.condition:
            before = job.config.model_dump(exclude={"lookahead"})
            after = config.model_dump(exclude={"lookahead"})
            job.config = config
            identity = self.identity(job)
            if before != after or (job.analysis_identity is not None and job.analysis_identity != identity):
                job.generation += 1
                job.chunks.clear()
                job.timings.clear()
                job.wall_seconds = job.wall_end = 0
                job.ignored.intersection_update(e['id'] for e in job.manual)
                job.confirmed.intersection_update(e['id'] for e in job.manual)
            job.analysis_identity = identity
            job.error = None
            job.active = True
            job.last_seen = time.monotonic()
            job.updated += 1
            self.persist(job)
            self.condition.notify_all()

    def identity(self, job):
        from provenance import file_hash
        return digest(dict(pipeline=pipeline_identity(getattr(self.models, 'root', None), job.config, self.worker_count),
                           certificate=file_hash(self.models.root/'heads'/f'{job.config.learned_heads}.json')
                           if job.config.learned_heads and getattr(self.models,'root',None) else None,
                           source=None if job.live else file_hash(job.path)))

    def refresh_identity(self, job):
        current = self.identity(job)
        if job.analysis_identity is not None and current != job.analysis_identity:
            job.generation += 1
            job.chunks.clear(); job.timings.clear()
            job.wall_seconds = job.wall_end = 0
            job.confirmed.intersection_update(e['id'] for e in job.manual)
            job.ignored.intersection_update(e['id'] for e in job.manual)
            job.updated += 1
            job.status = 'ready'
        job.analysis_identity = current
        return current

    def loop(self):
        while not self.closed:
            with self.condition:
                selected = None
                # Most recently controlled media gets priority.
                for job in sorted(self.jobs.values(), key=lambda j: j.last_seen, reverse=True):
                    if time.monotonic() - job.last_seen > 20 and not job.complete_scan:
                        continue
                    try:
                        self.refresh_identity(job)
                    except Exception as error:
                        job.error = f'Model identity error: {str(error)[:600]}'
                        job.status = 'error'; job.active = False; job.updated += 1
                        continue
                    index = job.next_chunk()
                    if index is not None:
                        selected = (job, index, job.generation, job.config.model_copy(deep=True), job.analysis_identity)
                        break
                if selected is None:
                    self.condition.wait(timeout=1)
                    continue
                job, index, generation, config, identity = selected
                job.running.add(index)
                job.status = "analyzing"
            started = time.perf_counter()
            try:
                analysis_path = job.path
                if job.live:
                    # A complete previous/current/next fragment gives both ASR and
                    # visual boundaries context without retaining decoded video.
                    with self.condition:
                        analysis_path = job.path.parent / f"analysis-window-{index}.mp4"
                        with analysis_path.open("wb") as output:
                            output.write((job.path.parent / "init.mp4").read_bytes())
                            for segment_index in (index - 1, index, index + 1):
                                fragment = job.segments.get(segment_index)
                                if fragment:
                                    output.write(fragment.read_bytes())
                try:
                    events = self.models.analyze(analysis_path, index * CHUNK, min(job.duration, (index + 1) * CHUNK),
                                                 job.duration, config, job.has_audio)
                finally:
                    if job.live:
                        analysis_path.unlink(missing_ok=True)
                with self.condition:
                    self.refresh_identity(job)
                    if job.generation == generation and job.id in self.jobs and job.analysis_identity == identity:
                        job.chunks[index] = events
                        finished = time.perf_counter()
                        job.timings.append(finished - started)
                        job.wall_seconds += max(0, finished - max(started, job.wall_end))
                        job.wall_end = finished
                        job.updated += 1
                        job.status = "analyzing" if job.running - {index} else "ready"
                        if not job.live and len(job.chunks) == math.ceil(job.duration / CHUNK):
                            job.status = "complete"
                            job.complete_scan = False
                        self.persist(job, index)
            except Exception as error:
                with self.condition:
                    if job.generation == generation:
                        job.error = f"{type(error).__name__}: {str(error)[:700]}"
                        job.status = "error"
                        job.active = False
                        job.updated += 1
            finally:
                with self.condition:
                    job.running.discard(index)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        for thread in self.threads:
            thread.join(timeout=5)
        if hasattr(self.models,'close'):
            self.models.close()


def re_full_id(value):
    return len(value) == 32 and all(c in "0123456789abcdef" for c in value)
