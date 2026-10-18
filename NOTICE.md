# Third-party components and model provenance

The root MIT license applies to code written for this project, not to third-party libraries, codecs, model weights or user media. Dependencies are installed separately; downloaded weights and runtime environments are excluded from the source archive.

## Licensed demonstration footage

The 60-second introduction uses **Anna Tarazevich / Pexels** kissing footage under the [Pexels License](https://www.pexels.com/license/) and prop/dialogue excerpts from **Tears of Steel (2012), (CC) Blender Foundation | [mango.blender.org](https://mango.blender.org/)** under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The GIF uses the Pexels source; the category-control poster is original project artwork under MIT and contains no third-party footage. The stock footage is not Creative Commons or MIT; film portions retain CC BY 3.0. The video includes aspect-preserving edits, captions, blur/concealment and real profanity with actual mute/beep output and an explicitly illustrative category-control collage. All timings are reviewed; the examples do not establish recognition accuracy. No synthetic voice, endorsement, complete source video, standalone soundtrack or private-library footage is included. [Full credits and changes](docs/MEDIA_CREDITS.md); [source and asset identities](media-assets.json).

| Component | Purpose | Upstream |
|---|---|---|
| faster-whisper / CTranslate2 | Local English transcription and word timestamps | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2) |
| Whisper tiny.en conversion | ASR weights | [Systran model](https://huggingface.co/Systran/faster-whisper-tiny.en), [Whisper](https://github.com/openai/whisper) |
| Whisper small.en conversion (optional) | Larger ASR weights | [Systran model](https://huggingface.co/Systran/faster-whisper-small.en) |
| NudeNet 3.4.2 | Anatomy detection; its package includes `320n.onnx` | [NudeNet package](https://pypi.org/project/nudenet/), [upstream repository](https://github.com/notAI-tech/NudeNet) |
| CLIP ViT-B/32 ONNX | Experimental image/text similarity | [ONNX model](https://huggingface.co/Xenova/clip-vit-base-patch32), [original CLIP](https://github.com/openai/CLIP) |
| ONNX Runtime | CPU / optional CUDA inference | [ONNX Runtime](https://github.com/microsoft/onnxruntime) |
| FFmpeg / libx264 | Media normalization, live segmentation and export | [FFmpeg licensing](https://ffmpeg.org/legal.html) |
| imageio-ffmpeg | Platform-specific FFmpeg executable | [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) |
| PyAV | Timestamp-aware media decoding | [PyAV](https://github.com/PyAV-Org/PyAV) |
| V-JEPA 2 ViT-L (optional) | Motion features for learned detectors | [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256), declares MIT |
| AST AudioSet (optional) | Audio-event features and violence candidates | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), declares BSD-3-Clause |
| SAM 2.1 tiny / Transformers / Torchvision (optional) | Offline prompted outline drafts for mask supervision | [Meta checkpoint](https://huggingface.co/facebook/sam2.1-hiera-tiny), [SAM2 upstream](https://github.com/facebookresearch/sam2), [Transformers image interface](https://huggingface.co/docs/transformers/model_doc/sam2) |
| PyTorch, Transformers, scikit-learn (training only) | Feature extraction, ONNX export, head training in `.train-venv` | [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [scikit-learn](https://github.com/scikit-learn/scikit-learn) |

CLIP downloads are pinned to revision `d15189d7028b43f1d3e65039190477f6af591c2a`. CPU mode uses quantized vision/text weights. Optional GPU mode can use the floating-point vision encoder. `optimize_clip.py` can produce a local equivalent patch-projection graph and records its deterministic output comparison. That transformed file is not redistributed here.

Whisper tiny.en downloads and offline loading use revision `0d3d19a32d3338f10357c0889762bd8d64bbdeba`. Both model revisions match the local performance and speech fixtures recorded with this release.

Optional small.en uses revision `d1d751a5f8271d482d14ca55d9e2deeebbae577f`. Its real-episode spot checks are separate from the original tiny.en synthetic fixture and throughput benchmark.

Whisper and CLIP upstream repositories publish MIT licenses. The V-JEPA 2 and AST weights are downloaded at setup, exported to local ONNX files and never redistributed here; heads trained on them stay in `.runtime/heads`. NudeNet's Python package declares MIT; the package's bundled model provenance and any underlying training/export obligations should be checked separately before redistributing a bundled binary. This project does not represent that a Python package license settles all model/data rights. NVIDIA runtime packages retain NVIDIA's terms. FFmpeg builds can include GPL components such as libx264; inspect the specific binary before redistribution.

Model outputs are heuristic evidence. None of these upstream models is presented here as a certified or fully validated personal content-safety classifier. CLIP's image-prompt similarity in particular is not a calibrated scene probability or a severity assessment.

## Optional local scene runtime

Qwen3-VL official GGUF weights are downloaded from [Qwen](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF), pinned to `1cd86afb9a95c410a6038ab3b40d8b578c892266`. The model repository declares Apache-2.0. Inference uses the official [llama.cpp](https://github.com/ggml-org/llama.cpp) Windows CUDA 13.4 build `b11146`; llama.cpp is MIT, with separate NVIDIA runtime terms. Exact binary SHA256 values are recorded in `setup_verifier.py` and the local installation manifest. No downloaded executable or weights are included in the source distribution.

Optional [Qwen3-VL 8B GGUF weights](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) use pinned revision `f982a07559d4a2f6c8744d840bf6fccab30eea96`, also declaring Apache-2.0. Model and projector checksums are pinned separately in `setup_verifier.py`.

Optional NudeNet `640m.onnx` comes from the upstream `v3.4-weights` release, asset `176832019` (103,538,690 bytes), SHA256 `04fe3d77980780c1f8297dc6d7f942fd5b3abe6942a188f742a85241e4f634eb`. The installer validates the checksum and loads the ONNX graph before accepting installation. An earlier download returned an HTML page; it was rejected by the inference test and replaced with the verified release binary.

SAM 2.1 tiny is installed explicitly at Meta revision `de431c4043854a71d8101e17995dfe596bf101a5`; its safetensors SHA256 is `48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`. Weights and private frames/masks are not distributed with project source. This optional data-preparation model determines prompted geometry, not whether depicted anatomy should be filtered. Exact downloaded artifacts retain their own upstream terms.
