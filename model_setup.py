"""Explicit, reproducible model download. Playback never downloads models."""
from pathlib import Path
import argparse
import os

CLIP_REPO = "Xenova/clip-vit-base-patch32"
CLIP_REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"
WHISPER_REVISION = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
SPEECH_REVISIONS = {
    "tiny.en": WHISPER_REVISION,
    "small.en": "d1d751a5f8271d482d14ca55d9e2deeebbae577f",
}


def runtime_dir():
    return Path(os.environ.get("ADAPTIVE_CENSOR_WORKDIR", Path(__file__).parent / ".runtime")).resolve()


def prepare_environment():
    root = runtime_dir()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    cuda_cache = root / "cuda-cache"
    cuda_cache.mkdir(exist_ok=True)
    os.environ.setdefault("CUDA_CACHE_PATH", str(cuda_cache))
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(512 * 1024**2))
    return root


def speech_model_path(root, *, model="tiny.en", local_files_only=True):
    from faster_whisper.utils import download_model
    if model not in SPEECH_REVISIONS:
        raise ValueError("Choose tiny.en or small.en for speech")
    return download_model(model, cache_dir=str(root / "models"),
                          revision=SPEECH_REVISIONS[model], local_files_only=local_files_only)


def download(gpu=False, speech_model="tiny.en"):
    root = prepare_environment()
    from huggingface_hub import snapshot_download
    clip = root / "clip"
    files = ["tokenizer.json", "preprocessor_config.json", "config.json", "README.md",
             "onnx/text_model_quantized.onnx", "onnx/vision_model_quantized.onnx"]
    if gpu:
        files.append("onnx/vision_model.onnx")
    snapshot_download(CLIP_REPO, revision=CLIP_REVISION, allow_patterns=files,
                      local_dir=str(clip), max_workers=2)
    if gpu:
        from optimize_clip import optimize
        optimized = clip / "onnx/vision_model_matmul.onnx"
        if not optimized.exists():
            print(optimize(clip / "onnx/vision_model.onnx", optimized), flush=True)
    speech_model_path(root, local_files_only=False)
    if speech_model != "tiny.en":
        speech_model_path(root, model=speech_model, local_files_only=False)
    print("Models ready:", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also download the floating-point GPU vision model")
    parser.add_argument("--speech-model", choices=list(SPEECH_REVISIONS), default="tiny.en",
                        help="Also install a stronger speech model; inference stays offline")
    args = parser.parse_args()
    download(args.gpu, args.speech_model)
