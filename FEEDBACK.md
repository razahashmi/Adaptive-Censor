# Playback feedback

Under the player, click **Wrongly censored** or **Missed something**. One click saves a report at the current playback time without pausing playback or changing filters. These controls are also available in fullscreen. The saved review window covers up to ten seconds before the click and two seconds after it, clamped to the file boundaries. It is approximate context, not a precisely labeled interval.

The acknowledgement offers **Undo report** and **Add details (optional)** for a category and short note. A failed save stays visibly unsaved and can be retried at its original time, even after switching media or reloading the page. The pending request is backed up in browser storage when available. A stable report UUID makes retries idempotent, including when a response was lost after the write. Restarted-server authentication is renewed once. A new unsaved report cannot be created until the pending report is saved.

Open **Saved reports** ([local review queue](http://127.0.0.1:8502/feedback.html)) to inspect pending, confirmed, dismissed and undone reports. **Review original (uncensored)** explicitly opens the saved source at the review window; it never autoplays. Categories and notes can be added later. Version checks reject conflicting edits from another tab. Confirming an issue is triage: it does not establish ground truth or train anything.

Optional **Keep a short local clip for later review** is off by default. It retains the report window in the background using CPU encoding, without model inference or GPU work. Saved clips can be replayed after the source session is removed. The queue supports keeping/retrying/removing clips and preparing precise human-label tasks. See [the complete foundation workflow](FOUNDATION.md).

## Storage and provenance

Reports are stored in `.runtime/feedback/reports.sqlite3`, separately from labeling projects, masks, model weights and evaluation runs. They survive server restarts and removal of the media session. Each record contains:

- A stable report ID, timestamps, report kind, playback time, optional category/note and review status.
- The viewer's policy and visual/audio action, plus its analysis generation and snapshot revision.
- Server-side title, source session reference, file size/modification time, analysis identity/configuration, nearby detections and the effective plan at receipt.
- Whether analysis changed between playback and receipt, whether that time had been analyzed, and live retention information. These flags prevent a disabled filter, review-only suggestion, buffering issue or changed model from being silently interpreted as a model mistake.

Saving report metadata performs no inference, frame decoding, encoding, full-media hashing or GPU work. Opt-in retention adds one bounded background CPU decode/encode, with at most eight queued/running requests and a 512 MiB clip collection limit. Retained clips have SHA256 identities. Original source size/modification time is a cheap availability check, not a whole-file checksum. Reported examples and titles used for model development must not later be presented as an untouched final test.

Without opt-in retention, keep the saved local media to review it later. Removed/changed files and expired live buffers are shown as unavailable; retaining live buffers is not supported yet. With retention, source removal is refused while its clip is queued/saving, and the finished clip survives later source removal. An interrupted capture can be retried while the source remains available. Explicitly removing a retained clip prevents duplicate report retries from recreating it; keeping it again is a separate action. Copies already made in labeling/training datasets remain independent.

If a report has never reached the server and its session is removed, its browser retry remains pending until that session is available; do not remove media while reports are unsaved. Existing saved reports can always be retried without duplicating them, even after the source is removed.

**Export reports** downloads JSONL metadata, excluding undone reports. It includes titles, notes, filter phrases/settings, detections and retained-clip/task status/identities, so review the file before sharing it. It contains no media, application tokens or live stream URL. All report endpoints use the same existing local session cookie and write-token protection as the player. This feature adds no cloud collection and does not change the app's loopback-only deployment boundary.

## Data workflow

1. Collect during normal playback with one-click reporting.
2. Review the original context and confirm or dismiss the issue. Separate model misses, wrong detections, localization/timing mistakes and policy or analysis-state problems.
3. Retain the clip, choose a scene category, confirm the issue and prepare a precise task with the actual series/movie title and clip-relative target bounds. The existing scene editor opens with empty labels in a development-only project. Review every category; neither a reported miss nor a false alarm supplies automatic positive/negative labels.
4. Explicitly admit completed human labels to the training collection only for titles permanently assigned to training, without held-out conflicts, under the current category definitions. Clip/task/label identities and prior model exposure are recorded. Language/custom reports remain in the report queue because the current scene editor supports five visual categories. Independently sampled evaluation data remains necessary: mistake reports alone cannot measure overall accuracy.
5. Retrain and compare the complete player pipeline before promoting a model. No report, including a confirmed issue, automatically changes training data, benchmark labels, model settings or censorship.

## Verification

`python -m unittest test_playback_feedback test_feedback_evidence test_readiness -v` covers report persistence, concurrent retries, clip retention/integrity/recovery, label admission, title conflicts, source deletion and setup diagnostics. `node --test web/feedback.test.mjs` exercises actual report/review handlers, including unsaved-edit protection and asynchronous clip status. The public-tree runtime suite completes 214 tests (seven skipped); the browser-module suite passes 53 tests as of 18 October 2026. Generated native-frame capture and extracted-package smoke tests verify the clip-to-label path without private media or model inference.
