# Hardware, downloads and compatibility

These are tested configurations and planning estimates, not certified minimums.

| Configuration | Status | Notes |
|---|---|---|
| Windows + 64-bit Python 3.13, CPU runtime | Locally tested | Base installation, generated controls, export/restart and short public diagnostic |
| AMD Ryzen 9 9950X3D, 30.9 GiB system RAM | Measured development machine | 26s public CPU diagnostic completed in 11.84s; peak process RSS 1229 MiB |
| NVIDIA speech/anatomy option | Experimental | Separate dependency setup; do not infer working CUDA from a listed provider |
| NVIDIA scene verifier / learned models | Experimental | Separate models, memory and validation; not covered by the CPU starter timing |
| Other Windows machines / fresh OS | Independent testing pending | Use the installation issue form and clean-install checklist |
| macOS / Linux / other Python versions | Not release-validated | Source may be adaptable; no packaged support claim |

The Windows setup shell uses the machine's installed .NET Framework 4.x and an existing 64-bit Python 3.13. It does not bundle Python, codecs, inference runtimes or model weights. Dependencies download from their registries; models download from the pinned upstream sources in NOTICE.md.

Reserve **several GiB** for the CPU environment and base models, plus space for source copies and exports. Optional scene/learned/segmentation models add several more GiB and may need substantial GPU memory. Exact downloads depend on options, wheels and caches. Keep at least enough additional storage for the original media plus a normalized copy and export. These estimates are not measured minimum RAM/VRAM requirements.

Use **Check setup** to inspect actual installed files, free disk space and package conflicts. For throughput comparisons, use the exact benchmark config and disclose concurrent workloads. A lookahead buffer handles temporary delays; sustained processing must keep pace.
