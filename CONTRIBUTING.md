# Contributing

This preview welcomes installation reports, reproducible bugs, accessibility improvements, performance work and careful detection research. Keep changes small enough to review and state their validation and limitations.

Start with the [architecture map](docs/ARCHITECTURE.md), [starter tasks](docs/STARTER_TASKS.md) and [public evaluation pack](samples/README.md). Use the issue forms to separate bugs, installation reports, detection errors and model proposals.

## Development checks

Install the pinned runtime requirements in a virtual environment. Model downloads are separate and are not needed for the generated-media tests. With that environment active:

```powershell
python -m unittest discover -p 'test_*.py' -q
$webTests = @(Get-ChildItem web -Filter '*.test.mjs' | ForEach-Object FullName)
node --test @webTests
python build_source_release.py --output dist
python release_smoke.py dist/adaptive-censor-0.3.0-preview.1-source.zip
```

Node is needed for tests, not to run the player. Training-dependent tests may skip in the runtime-only environment. Install the separate training dependencies only for relevant model changes; report skips rather than claiming full training coverage. CI uses an isolated Windows CPU environment and generated demo media.

## Useful bug reports

Include the application version, operating system, Python/browser version, CPU/GPU type, selected settings, steps to reproduce and expected/actual behavior. Scrub paths, titles, speech text, tokens and other private information from logs. Use generated or explicitly redistributable media. Do not upload unlicensed film/TV scenes, private recordings, model weights, labels sourced from private media, or whole runtime folders.

Public media contributions require documented permission to redistribute and adapt the example, with its creator, source, license and changes recorded. A citation by itself does not satisfy this contribution policy. The existing demo is a narrow exception for three reviewed assets with separately documented MIT graphics, Pexels and CC BY 3.0 source terms; replacing it requires updating `docs/MEDIA_CREDITS.md`, `media-assets.json` and the reviewed packaging checks. The complete source files remain outside the repository. Do not add website logos, unrelated music, fonts or other third-party files without their own appropriate permission and notices.

The player's **Wrongly censored** / **Missed something** controls are intended for local collection. Exported reports are not automatically anonymous and are not ground truth. Review and precisely label content before using it to train.

## Detection research

State which categories and operating points changed. Preserve permanent title-level train/validation/test assignments. Never train on evaluation titles, count model drafts as independent human labels, or equate mostly clean frame accuracy with detection quality. Report per-category time precision/recall, event/title support, false-filter time and clean-window false alarms. Measure intended-pixel coverage separately from interval coverage. Include complete player-pipeline speed and resource costs for a proposed default change.

Do not enable experimental thresholds merely to obtain more detections. Qualification requires independent human evidence for the exact model and configuration. Source-control checks, unit tests and a generated demo do not establish model accuracy.

Contributions to original project code are submitted under the project's MIT license. Preserve upstream notices and identify any new third-party dependency, binary, model or data terms. Report security concerns according to [SECURITY.md](SECURITY.md).
