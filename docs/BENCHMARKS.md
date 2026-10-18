# Reproducible benchmark and evidence scorecard

This page distinguishes **rendering tests**, **draft development diagnostics** and **independent recognition evaluation**. They are not interchangeable.

## Public CPU diagnostic — 18 October 2026

The starter has five clips, 26 seconds, from **two source titles already inspected during development**. Labels are assistant drafts, not independent human ground truth. Use this run to reproduce pipeline behavior and investigate failures, not to claim population accuracy.

| Measurement | Local result |
|---|---:|
| Mode | CPU speech + anatomy + CLIP review suggestions |
| CPU | AMD Ryzen 9 9950X3D, 16 cores / 32 logical cores |
| Media / wall time, including cold start | 26.00s / 11.84s |
| Media seconds per wall second | 2.20× |
| Cold first chunk | 4.57s |
| Subsequent chunk p95 | 1.70s |
| Peak process memory | 1229 MiB |
| Sustained live / multi-user capacity | Not tested |

This is a short CPU run on one existing machine, not a minimum hardware requirement or a general real-time promise. [Exact config, dependency versions, model hashes and case results](results/public-starter-cpu.json).

| Category | Draft reference support | Automatic output in this run | Interpretation |
|---|---|---|---|
| Kissing | 2 positive intervals, 2.41s | 0s accepted filtering | Review-mode suggestions are not automatically accepted. This is not a qualified kissing detector. |
| Language | 1 abusive word; 1 clean dialogue clip | 0.28s matched / 0.84s filtered | Reference-time recall 100%, precision 33.3%; timing padding counts as extra filtering. One word cannot establish speech accuracy. |
| Anatomy / sexual activity / violence / gore | No reviewed positive intervals | No incorrect filtering in the tiny reviewed negative windows | Positive recall is unmeasured, not 100%. The action interval and organ-prop gore semantics remain unknown. |
| Pixel-mask coverage | No independently reviewed masks | Unmeasured | Temporal coverage does not establish which pixels are covered. |

The JSON reports reviewed seconds, misses, unnecessary-filter seconds, false-filter seconds per clean hour, title support and explicit missing evidence. Small-denominator normalized rates are unstable; retain raw seconds and event counts.

## Reproduce and improve

Follow [the starter-pack guide](../samples/README.md). `public_benchmark.py prepare` verifies full-source hashes before making normalized clips. `run` uses the same `Models.analyze` passes as the player and records cold/warm timing and policy behavior. **Draft scoring fails unless `--allow-draft` is explicit.**

Before making an accuracy claim: obtain human-reviewed exact intervals, add positive examples for absent categories, keep new evaluation titles out of training/model selection, and report title-level uncertainty. Use `evaluate_labels.py` for complete human projects and `evaluate_masks.py` for independent human pixel masks. The public starter is development data even after review.

## Existing private development evidence

[LIMITATIONS.md](../LIMITATIONS.md) describes the prior small development comparison (42 reviewed clips; nudity precision 35.3%, recall 31.9%, with three positive events across two titles). That sparse, repeatedly inspected evidence does not qualify a detector. Private media and labels are not included in public source or the starter pack.

## Validation status

- Renderer / policy conformance: automated tests and reviewed visual/audio examples.
- Public reference diagnostics: reproducible CPU report above, assistant labels disclosed.
- Independent public recognition benchmark: **not established**.
- Sustained live, GPU variants and clean-OS installation: separate verification required.
