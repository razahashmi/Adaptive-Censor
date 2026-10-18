# Adaptive Censor 0.3.0-preview.1

The first public experimental preview of a local video and podcast player with analysis ahead of playback, adjustable picture/sound rules and a timeline you can correct.

## Try it

**Windows + 64-bit Python 3.13** is the tested base. Download and extract the complete Windows setup ZIP, open **Adaptive Censor Setup.exe**, then select **Set up** and **Open player**. Start with the controls demo before adding the separate models. The helper is unsigned and downloads dependencies separately; independent clean-Windows and full native-GUI testing are still pending.

For source setup, run `Setup.ps1 -SkipModels` and `Start.ps1`, then open `http://127.0.0.1:8502/`. See [installation](https://github.com/razahashmi/Adaptive-Censor/blob/main/docs/INSTALLATION.md).

## Included

- Sliding lookahead with a default 30-second analysis target, six-second chunks, cached results and playback waiting at unfinished coverage.
- Separate category picture and sound rules: adjustable blur, opaque concealment, mute and beep.
- Local English word/phrase filtering, regional anatomy boxes and experimental scene analysis.
- Side-by-side viewing, editable intervals, manual regions, saved sessions and filtered MP4 export.
- One-click local mistake reports, optional retained context and explicit human-review/data tools. Reports never upload or train automatically.
- A 63-second licensed demo with actual rendering and edited authentic UI captures, captions, contributor guides, a model card and reproducible public diagnostics.

## Evidence and limits

Local checks completed **223 runtime tests with seven optional skips**, and **53 browser tests passed**. Fresh CPU setup, archive startup, feedback, export and restart checks passed. These are software checks, not recognition accuracy measurements.

The five-clip public CPU diagnostic processed 26 seconds in 11.84 seconds including startup on one Ryzen 9 9950X3D. Its references are drafts and its sources were already inspected. It does not establish representative accuracy or sustained live capacity. [Exact configuration and results](https://github.com/razahashmi/Adaptive-Censor/blob/main/docs/BENCHMARKS.md).

Detection remains experimental and can miss content or filter clean scenes. Scene suggestions require confirmation by default. This is not a validated child-safety product and no 95% accuracy claim is supported. Audio replacement affects the full mix; regional boxes can miss intended pixels. GPU, segmentation, learned models and live input need broader testing. Household streaming is planned; the preview listens locally.

## Download contents

| Asset | Purpose |
|---|---|
| `adaptive-censor-0.3.0-preview.1-windows-setup.zip` | Native setup helper, its verified source ZIP, license and build receipt; Python 3.13 required |
| `adaptive-censor-0.3.0-preview.1-source.zip` | Audited, allowlisted source with per-file hashes and no private development history |
| `adaptive-censor-demo.mp4` | The 63-second introduction; reviewed effects, staged correction and compact end credits |
| `adaptive-censor-public-examples.zip` | Optional five-clip licensed development pack with draft annotations |
| `adaptive-censor-demo-ui-inputs.zip` | Optional authentic UI screenshot inputs for reproducing the demo |
| `PREVIEW_VALIDATION.json` and `ASSET_SHA256SUMS` | Validation scope and checksums for these release assets |

The source/archive excludes private media, private datasets and labels, downloaded weights and runtime environments. Original code is MIT; dependencies/models and media keep separate terms. The demo uses Anna Tarazevich / Pexels footage and *Tears of Steel* / Blender Foundation footage under CC BY 3.0. [Credits and changes](https://github.com/razahashmi/Adaptive-Censor/blob/main/docs/MEDIA_CREDITS.md).

[Project write-up](https://github.com/razahashmi/Adaptive-Censor/blob/main/docs/PROJECT_WRITEUP.md) · [Contribute](https://github.com/razahashmi/Adaptive-Censor/blob/main/CONTRIBUTING.md) · [Starter tasks](https://github.com/razahashmi/Adaptive-Censor/blob/main/docs/STARTER_TASKS.md) · [Known limitations](https://github.com/razahashmi/Adaptive-Censor/blob/main/LIMITATIONS.md).
