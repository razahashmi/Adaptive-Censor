# Model and policy card — 0.3 preview

## Intended use

Local, preference-driven media playback with reviewable filtering. Users can correct category, interval and region decisions. This preview is not independently validated for unattended or child-safety use.

## Components and defaults

| Component | Role | Default / requirement |
|---|---|---|
| faster-whisper tiny.en | English words and timestamps | Base speech; small.en is optional |
| NudeNet 320n | Exposed-anatomy boxes | Base anatomy; boxes do not establish sexual context |
| CLIP ViT-B/32 ONNX | Image/text similarity candidates | Suggestions require review; similarity is not a probability |
| Qwen3-VL local verifier | Short-sequence scene proposals/checks | Optional NVIDIA setup; scene suggestions require confirmation by default |
| Learned heads / V-JEPA / AST | Research scene/motion/audio features | Optional; exact configuration needs independent qualification |
| SAM / mask tools | Offline prompted draft outlines | Research data tools, not a validated universal playback segmenter |

Exact model revisions, download checksums and separate licenses are in [NOTICE.md](../NOTICE.md), `model_setup.py` and `setup_verifier.py`. Weights are not bundled in source or the setup launcher.

## Filtering semantics

Anatomy/nudity, sexual activity, kissing, violence, gore and language are separate categories. Ordinary kissing is not automatically sexual activity. Allow/blur/conceal and keep/beep/mute are independent rules. Opaque concealment covers only accepted intervals and selected regions. Muting affects the full mix, not one isolated voice. Blur strength does not improve recognition.

## Evidence and limits

The existing small private development comparison is described in [LIMITATIONS.md](../LIMITATIONS.md). It is not an independent public benchmark. The public starter contains already inspected examples with **assistant-drafted** reference labels; its results are diagnostic until human-reviewed and remain development-only afterward. See [the scorecard](BENCHMARKS.md).

No new 95% precision/recall claim, automatic detector promotion or population-level reliability follows from the demo, CI tests, report count, or the starter pack. Missing categories and pixel masks must remain marked unmeasured. Speech, accents, brief shots, motion, unusual framing and confusing clean scenes can fail.

## Improving a model

Collect precise human annotations with consent/redistribution rights; keep whole titles separate across training and evaluation; publish event/time/pixel evidence, false-filter cost and full-pipeline throughput. Match the exact model and analysis settings. Reports do not train or activate a model automatically.
