# Preview validation status · 18 October 2026

Publication destination: **https://github.com/razahashmi/Adaptive-Censor**, preview tag `v0.3.0-preview.1`. This release is prepared from an audited public copy, preserving the repository's initial license/README commit and adding no private development history. Hosted CI status is available in [Actions](https://github.com/razahashmi/Adaptive-Censor/actions/workflows/ci.yml); consult the actual run for its result.

## Verified locally

- Windows, 64-bit Python 3.13.9, CPU dependencies in an isolated environment. **223 runtime tests completed successfully, with seven optional training-dependency skips**. All **53 browser-module tests** passed. Skips do not establish training/GPU compatibility.
- The native Windows .NET setup helper compiled and passed a headless checksum/Python self-test. It requires existing Python 3.13, downloads dependencies/models separately, is unsigned, and bundles no third-party runtime. GUI operation and independent clean-Windows installation are not established by that test.
- The source builder uses an explicit allowlist, checks Python syntax, audits personal paths/common credentials, rejects links/private runtime paths, and binds three licensed binary showcase assets to exact hashes and individual source terms. Private media, datasets, downloaded weights, runtime environments and development Git history are excluded. The public starter's draft annotation manifest and diagnostic report are intentional public research examples, not private labels.
- The final showcase is **63 seconds, 1,512 frames, one picture track and one audio track**. All 144 kissing-example frames and 96 concealment-excerpt frames were checked; concealment is opaque and clean intervals change only within codec tolerance. Captions fit the frame. Audio is original at 0:29, beeped at 0:32, muted at 0:35, and silent outside that chapter. The poster is original MIT artwork. Pexels and film portions retain separate terms.
- Authentic edited player screenshots demonstrate actual controls. The correction chapter explicitly stages a false match on a manual timeline, then performs the real report/exclude workflow. It is not a recognition or speed test.
- The public CPU diagnostic processes **26 seconds in 11.84 seconds**, including startup, on one Ryzen 9 9950X3D machine. Draft references, precise configuration and model/dependency identities are disclosed. This tiny development diagnostic does not establish representative accuracy or sustained live throughput.
- Earlier extracted-archive smoke tests covered local authentication, generated playback, feedback save/retry/review/withdrawal/export, optional clip retention, guarded human-label ingress, filtered export and restart persistence. Final artifact-specific smoke, setup and cleanup receipts accompany the locally built release packages; inspect their source checksum when verifying a particular archive.

## Still pending

- A successful hosted Windows CI result must be checked in Actions. The workflow builds source/native artifacts and checks their extracted behavior; local tests alone do not establish a hosted pass.
- Independent testing on a clean Windows OS, first-time model downloads and the full native setup interaction. A new virtual environment on the development machine is not a clean OS.
- Optional GPU/training installations, other operating systems/Python versions, long-run recovery, and sustained live or multi-user capacity.
- Independent category-level recognition evaluation, moving-pixel coverage, detector qualification or a 95% accuracy result. No new detector is promoted by these packaging/documentation changes.

Release scope: an experimental preview for people willing to review and correct timelines. See [limitations](LIMITATIONS.md), [model card](docs/MODEL_CARD.md), [benchmark evidence](docs/BENCHMARKS.md) and [clean-install checklist](docs/CLEAN_INSTALL_CHECKLIST.md).
