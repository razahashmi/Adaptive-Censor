"""Limited local detectors. No claim of general scene understanding."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

NUDITY_CLASSES = {"FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
                  "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"}


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold(), flags=re.UNICODE)


def matches(text: str, phrases: list[str]) -> bool:
    words = tokens(text)
    return any(needle and any(words[i:i + len(needle)] == needle
                             for i in range(len(words) - len(needle) + 1))
               for needle in map(tokens, phrases))


def scan_srt(text: str, phrases: list[str], duration: float) -> list[dict]:
    """Whole-cue redaction; subtitle timings are NOT word timings."""
    timestamp = r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})"
    pattern = re.compile(timestamp + r"\s*-->\s*" + timestamp)
    events = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        match = pattern.search(block)
        if not match or not matches(block[match.end():], phrases):
            continue
        values = list(map(int, match.groups()))
        time = lambda v: v[0] * 3600 + v[1] * 60 + v[2] + v[3] / 1000
        start, end = max(0, time(values[:4])), min(duration, time(values[4:]))
        if end > start:
            events.append({"start": start, "end": end, "category": "language", "confidence": 1,
                           "severity": 2, "enabled": True, "source": "subtitle keyword",
                           "note": "Whole subtitle cue matched; timing is not word-aligned."})
    return events


@lru_cache(maxsize=1)
def speech_model(model: str, cache: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model, device="cpu", compute_type="int8", download_root=cache)


def scan_speech(path: Path, phrases: list[str], cache: Path, model="tiny.en", progress=None) -> list[dict]:
    segments, info = speech_model(model, str(cache)).transcribe(
        str(path), word_timestamps=True, vad_filter=True, beam_size=5,
    )
    words = []
    for segment in segments:
        for word in segment.words or []:
            for token in tokens(word.word):
                words.append((token, word.start, word.end, word.probability))
        if progress:
            progress(min(.99, segment.end / max(info.duration, .001)))
    events = []
    for phrase in phrases:
        needle = tokens(phrase)
        if not needle:
            continue
        for i in range(len(words) - len(needle) + 1):
            span = words[i:i + len(needle)]
            if [w[0] for w in span] != needle:
                continue
            # Prevent multiword matches from crossing long unrelated pauses.
            if any(b[1] - a[2] > 1 for a, b in zip(span, span[1:])):
                continue
            if span[-1][2] <= span[0][1]:
                continue
            events.append({"start": span[0][1], "end": span[-1][2], "category": "language",
                           "confidence": min(w[3] for w in span), "severity": 2,
                           "enabled": True, "source": "Whisper keyword",
                           "note": "Configured phrase matched. Score is ASR confidence, not severity."})
    if progress:
        progress(1.)
    return events


@lru_cache(maxsize=1)
def nudity_model():
    from nudenet import NudeDetector
    return NudeDetector(providers=["CPUExecutionProvider"])


def scan_nudity(path: Path, duration: float, fps: float = 2, threshold: float = .5,
                 progress=None) -> list[dict]:
    """Sparse image sampling detects exposed anatomy, not sex, kissing, or violence."""
    import cv2
    detector = nudity_model()
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError("The visual detector could not decode this video.")
    interval = 1 / fps
    events = []
    index = 0
    try:
        while index * interval < duration:
            t = index * interval
            video.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = video.read()
            if not ok:
                raise RuntimeError(f"Visual decoding failed at {t:.2f}s; scan is incomplete.")
            hits = [hit for hit in detector.detect(frame)
                    if hit["class"] in NUDITY_CLASSES and hit["score"] >= threshold]
            if hits:
                # Expand sampled evidence by one interval on each side before user padding.
                start, end = max(0., t - interval), min(duration, t + interval)
                score = max(float(hit["score"]) for hit in hits)
                if events and start <= events[-1]["end"]:
                    events[-1]["end"] = end
                    # Use the weakest sampled positive score for conservative UI thresholding.
                    events[-1]["confidence"] = min(events[-1]["confidence"], score)
                else:
                    events.append({"start": start, "end": end, "category": "nudity",
                                   "confidence": score, "severity": 2, "enabled": True,
                                   "source": "NudeNet sampled frames",
                                   "note": "Exposed-anatomy candidate; severity is an editable placeholder."})
            index += 1
            if progress:
                progress(min(1., index * interval / duration))
    finally:
        video.release()
    return events
