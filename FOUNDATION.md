# Foundation work · 15 October 2026

The first complete path from a playback mistake to reviewed training data is implemented. It does not train or enable a detector automatically and does not change detection thresholds, tracking or accuracy claims.

## Report, retain, label, then admit to training

1. Click **Wrongly censored** or **Missed something**. Optional **Keep a short local clip** is off by default. The report saves immediately; a single background CPU encoder retains up to 12 seconds, using two encoder threads. There is no model inference or full-film hashing in reporting.
2. In Saved reports, keep/retry/remove the clip and review it explicitly. Saved clips survive removal of the original media session. Retention has a 512 MiB collection limit and at most eight queued/running requests. The original session cannot be removed while its clip is queued or saving. Interrupted capture remains retryable; changed sources/clips are rejected.
3. Confirm the issue, choose a scene category and open **Prepare precise scene labels**. Enter the actual series/movie title, select a clip-relative target and create the task. This copies an integrity-checked clip into `feedback-review-v1` with **empty labels**, a development split, recorded model exposure and no evaluation eligibility. Language/custom reports remain in the report queue because this editor supports the five existing scene definitions.
4. Review all scene categories in the existing label editor. The server records prior model exposure even if the browser omits that flag. Changed/missing labeling clips cannot be served or relabeled. Original report windows are approximate context, never automatically positive or negative labels.
5. **Add completed human labels to training** is a separate explicit action. It requires completed human labels, the source/clip/task identities, a title permanently assigned to train in `labeling/library.json`, and no validation/test title conflict in existing projects. Unknown, development-only, validation and test assignments are refused. Copied examples enter `training-feedback-human-v1`; no training, model replacement or automatic filtering is triggered.

Title grouping depends on the reviewer supplying the real source title. The checks preserve known assignments and conflicts; they cannot establish the identity of an intentionally mislabeled movie. Retained clips have SHA256 identities; the original whole recording is identified by its session and size/modification metadata, not a full content hash.

Retention, review tasks and admitted training examples are distinct local copies. Removing retained evidence does not erase a clip already copied into a labeling dataset. Undo marks the report withdrawn; it does not erase retained evidence or dataset copies. The queue offers explicit retained-clip removal. Exports remain metadata only. No uploads are automatic.

## Preserve work and recover visibly

Unsaved report-review edits and prepared label boundaries block report navigation/refresh until saved, submitted or explicitly discarded. Capture-status polling preserves edits and does not autoplay or reveal the original. Duplicate report/capture/task/training submissions are idempotent; conflicting edits and changed dataset entries are refused. The new optional retention field preserves fingerprints of reports queued by the older app.

## Installation checks

The player links to **Check setup** at `/readiness.html`. It checks pinned direct dependencies, executable availability, writable storage/free space, base model files and installed ONNX provider options. It performs no inference, download, media reading or GPU execution test. GPU ONNX Runtime may supply the runtime module without the separate CPU distribution. Findings explain corrective actions without exposing private filesystem paths.

The check also reports overlapping CPU/GPU ONNX packages, exact mismatched versions and the loaded module version. Setup removes overlapping distributions before installing the selected runtime; the shared module files otherwise make switching back from GPU to CPU unreliable. Stop the running service before repairing its environment.

`python readiness.py` provides the same local diagnostic. `Setup.ps1 -SkipModels` installs dependencies and skips separate model downloads so the generated controls demo can be tried first; it cannot be combined with explicit scene/segmentation model downloads. Normal setup remains available for analysis.

## Next accuracy gate

Use the new human review/training path to collect genuine misses and confusing negatives across assigned training titles. Then compare a temporal region/tracking candidate against a frozen baseline on separate human-reviewed titles, including intended-pixel coverage, false-filter seconds, missed seconds, correction effort and sustained speed. Object tracking and detector quality remain unqualified; this foundation work makes improvements testable and recoverable.
