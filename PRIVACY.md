# Privacy in the local preview

The application performs media analysis locally. It has no application telemetry, automatic feedback upload or cloud inference. Setup downloads dependencies and model weights from their upstream hosts. Optional live input contacts its URL and the media/playlist hosts it references; those hosts receive the network requests.

Imported files are copied and normalized into `.runtime/sessions-v2`. Detections, speech text, timeline edits, exports, model caches and logs can remain under `.runtime`. The **Remove local copy** action removes that session's app copy and analysis; it does not delete the original file. Logs, interrupted live buffers, exports and separate research/feedback collections may require separate removal. An alternate `ADAPTIVE_CENSOR_WORKDIR` changes the storage location.

Playback reports are saved in `.runtime/feedback/reports.sqlite3`. They include the media title, playback time, settings (including configured filter phrases), nearby detections and optional notes. An unsaved report is backed up in the same browser's local storage when available. Reports are not training labels and survive removal of the media session. Video retention is separately opt-in: **Keep a short local clip** preserves up to 12 seconds under `.runtime/feedback/evidence`, with a 512 MiB collection limit. Clips remain on this server. **Undo report** marks the report withdrawn; it does not erase the database record or retained clip. The review queue offers explicit retained-clip removal; permanent report-record deletion is not yet provided.

Creating a precise labeling task explicitly copies the retained clip into a development labeling project. Admitting completed human labels to the training collection is another explicit copy. Removing the original media or retained evidence does not delete those labeling/training copies. These examples record prior model exposure and are not independent evaluation material. See FOUNDATION.md for the title-split checks and copy lifecycle.

Reports can be exported deliberately as JSONL. Withdrawn reports are excluded. The exported metadata can contain private titles, notes, phrases and detected speech; review it before sharing. Nothing is sent to maintainers automatically. Do not attach raw runtime folders or private reports to public issues.

Profiles and rules are remembered in browser storage. Research labeling tools retain annotation projects and, when explicitly prepared, clips, frames and masks under the runtime folder. Model suggestions, reviewed labels and feedback are distinct collections.

The server listens on `127.0.0.1`. Local session cookies, write tokens, host/origin checks and browser security headers are present. This preview is not intended for public exposure or multi-user hosting. Use normal operating-system access controls to protect the runtime directory and remove sensitive data you no longer need.
