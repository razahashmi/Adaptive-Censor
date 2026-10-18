# User guide

[Quick start and project introduction](../README.md).

## Make filtering fit your preferences

- **Choose the effect.** Use light or strong blur, opaque concealment, muting or beeping. Picture and sound rules are independent.
- **Choose the categories.** Configure anatomy, sexual activity, violence, gore, kissing, language and a custom visual concept separately. Kissing is separate from sexual activity.
- **Inspect the result.** Compare original and filtered playback side by side. Exclude false matches, confirm suggestions and add missed intervals.
- **Keep useful context.** Regional anatomy blur and manually selected regions can preserve the rest of the image. Automatic segmentation remains a research tool.
- **Save your work.** Reopen local sessions or export a filtered MP4. Changing playback rules does not require re-encoding the source.
- **Report mistakes while watching.** One click records a false blur or a missed moment. Optional local clips preserve context for later review.

Analysis runs on this computer. Ordinary local-file use needs no cloud inference subscription or API key. Installing dependencies and downloading models requires an internet connection.

## What can it filter?

| Category | What the rule is for | Current approach and limits |
|---|---|---|
| Anatomy / nudity | Exposed anatomy selected by the detector | Local anatomy detection with nearby-frame support and regional boxes; false positives and incomplete coverage remain possible. |
| Sexual activity | Sexual behavior, separately from kissing | Experimental scene verification or optional learned models; scene suggestions require confirmation by default. |
| Violence | Fighting, attacks and violent actions | Short visual sequences and optional audio evidence; the presence of a weapon alone does not establish violence. |
| Gore | Graphic injury and exposed organs | Experimental scene recognition; review timing and coverage before playback. The public demonstration uses a fictional brain prop. |
| Kissing | Mouth-to-mouth kissing | A separate category, so users can allow it while filtering sexual activity. |
| Language | User-selected spoken words or short phrases | Local English speech recognition, word timestamps, a phrase list and exceptions. |
| Custom | A visual concept the user describes | Experimental suggestions for review; arbitrary concepts are not reliably recognized. |

These are supported **categories and controls**, not a promise to recognize every example. Each picture category can be allowed, blurred at a chosen strength, or concealed. Sound can independently stay on, be muted or be replaced by a beep. Confidence thresholds and timing margins help tune behavior; increasing blur strength changes the effect, not recognition accuracy.

## How adaptive video filtering works

1. **Analyze locally.** The engine processes the file in small chunks, sampling frames and short native-frame sequences. Anatomy checks use nearby-frame support; optional scene verification examines temporal context. Model availability and selected analysis mode determine the passes that run.
2. **Build a reviewable timeline.** Results carry a category, time interval, confidence and, when available, a region. Inspect suggestions, confirm them, remove false matches and add missed moments.
3. **Apply your rules.** The player combines accepted intervals with independent picture and sound settings. For example, allow kissing, strongly blur sexual activity and conceal gore, while beeping selected language. Changes to playback rules do not require re-encoding the source or rerunning recognition; changing analysis settings may require another scan.
4. **Keep the rest visible.** Use regional anatomy filtering or a manually selected rectangle where appropriate. Full-scene actions cover the whole picture. A region must cover the intended pixels throughout the interval; general automatic object segmentation is still research work.
5. **Watch or export.** Inspect the side-by-side view, save a local session, or create a filtered MP4. Opaque concealment hides the selected area; even strong blur may leave recognizable shapes.

Here, **adaptive means configurable behavior and reviewable decisions**. It does not mean that the app silently learns your preferences or becomes more accurate after every report.

For incremental and experimental live playback, the engine targets a configurable lookahead, normally 30 seconds. The player waits if required analysis is unfinished. A 30-second buffer handles temporary slowdowns; sustained analysis must still keep pace with incoming video. There is no measured universal real-time guarantee for this release.

## Words, podcasts and sound

For videos and audio-only podcasts, local speech recognition estimates word boundaries. Your word/phrase list selects matches, and an exception list reduces unwanted matches. The language rule can keep sound, mute a matching interval or replace it with a beep. Review timing and adjust the margins to avoid exposing the beginning or end of a word.

Sound rules can also apply to reviewed visual events: for example, conceal a sexual-activity interval and mute its audio, or blur a violent scene while keeping the sound. A manually added interval can mute or beep an unwanted sound without speech recognition.

**Muting and beeping affect the full audio mix during the selected interval.** They do not isolate a speaker or preserve background music under a removed word. The base speech models are English; accents, overlapping voices, lyrics, music and unclear speech can cause misses or wrong timestamps. Optional audio models contribute evidence to experimental scene models; this is not a general automatic filter for screams, gunshots, sexual sounds or every other sound category. The public video demonstrates real mute/beep output on reviewed word timings. The original line contains the abusive word “dick”; the filtered versions beep or mute it. This does not establish speech-recognition accuracy.

## Using it around the home

**Available now:** run analysis and review on one computer, export the filtered MP4, then play that file on other devices or through your existing home media server. The original stays separate. Each export reflects the settings used to create it; another device does not receive the interactive filtering or feedback controls through that MP4.

**Planned server design:** a home machine would analyze a source once, cache the event timeline, and serve controlled playback to authenticated phone, browser and TV clients. Household profiles could use the same analysis with different picture and sound rules. A server-rendered stream would help devices that cannot apply the effects themselves.

That needs authenticated client access, profile permissions, secure streaming, device compatibility, cache/model versioning, a bounded job queue and measured capacity for simultaneous viewers. Those are development goals. **This preview binds to `127.0.0.1` and is not a supported LAN or multi-user server.** Changing its listening address alone would not provide those features.

A router cannot apply visual filters to every app: encrypted connections and protected streaming players do not expose decoded frames to it. Universal Netflix/Prime/other DRM-service filtering is outside this release. Supported future clients or permitted unprotected media streams would have to route playback through the censoring system.

## Quick start

The tested base configuration is **Windows + Python 3.13**. The optional Windows setup helper installs the CPU source preview; see [installation](INSTALLATION.md). Other systems and Python versions need wider testing.

Download the source preview, extract it into a writable folder, and open PowerShell there:

```powershell
.\Setup.ps1
.\Start.ps1
```

Open **http://127.0.0.1:8502/**. Stop the background service with `Stop.ps1`.

Setup creates an isolated Python environment, installs pinned dependencies and downloads the base English speech/CLIP models. The large optional scene verifier is a separate download.

For a controls-only first try, `Setup.ps1 -SkipModels` skips separate model downloads. The generated controls demo verifies the player, not recognition. The licensed-footage walkthrough above is a separate technical showcase.

If PowerShell blocks scripts, the equivalent base installation is:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu-win-py313.lock.txt
.\.venv\Scripts\python.exe model_setup.py
.\.venv\Scripts\python.exe server.py --port 8502
```

Open **Check setup** in the player to inspect installed files, dependency versions and storage. It does not run an accuracy test.

### Optional NVIDIA features

`Setup.ps1 -Gpu` installs optional speech/anatomy CUDA support. `Setup.ps1 -SceneVerifier` downloads the separate local vision-language verifier, requiring an NVIDIA GPU and several additional gigabytes. These are different options; their compatibility still needs broader independent testing. Stop the service before changing its dependency environment.

## From an imperfect prediction to a better model

1. **Watch and review.** Import your own local media, choose rules, analyze it and correct the timeline.
2. **Report the mistake.** Use **Wrongly censored** or **Missed something**, including in fullscreen. Optionally keep a short local evidence clip.
3. **Label precisely.** In Saved reports, review the context and create a task with the actual source title and exact boundaries. Labels start empty.
4. **Evaluate the improvement.** Completed human labels can enter a guarded training collection. Contributors retrain and compare against separate evaluation titles before proposing a model change.

**A click does not teach the model automatically.** Reports stay local and are not sent to maintainers. Confirmed reports are not ground truth; careful labeling and independent evaluation remain necessary. [Feedback workflow](../FEEDBACK.md) · [Training safeguards](../FOUNDATION.md)

## What this preview supports

| Available now | Still experimental | Outside this release's scope |
|---|---|---|
| Local-file player, reviewable timeline, side-by-side comparison, export and feedback | Scene recognition, learned classifiers, GPU variants, live input and segmentation tools | Protected streaming services, universal household filtering and reliable unattended censorship |

Scene suggestions require confirmation by default. Anatomy can filter automatically and can be wrong. Blur can preserve visible shapes; opaque concealment covers the selected area only during the specified intervals. A completed scan does not guarantee complete detection. [Read the limitations](../LIMITATIONS.md).

## Help build it

Contributions are welcome from developers, researchers and everyday testers:

- **Try a clean installation** and report reproducible bugs or hardware compatibility results.
- **Improve the player** through accessibility, clearer controls, reliable recovery and measured performance work.
- **Improve detection** with carefully reviewed examples, stronger models and complete comparisons of misses, false filtering, pixel coverage and speed.
- **Keep evaluations honest** by separating training and evaluation titles and reporting category-level results.

The highest-value improvements are **better human-reviewed data and honest evaluation**: include clear positives, confusing clean examples, exact event boundaries and region coverage; keep entire evaluation titles out of training. Report category-level precision, recall, false-filter time, missed time and end-to-end speed. For performance, reuse decoded frames and cached features, batch model work, and reserve expensive verification for uncertain candidates where evidence supports it. Model changes should earn promotion through independent evaluation, not simply detect more scenes.

Start with [CONTRIBUTING.md](../CONTRIBUTING.md). Use generated or explicitly redistributable examples for public reports. **Do not upload commercial film/TV clips, private recordings, transcripts, raw feedback or labels derived from private media.**

## Privacy

The app binds to this computer's loopback address. There is no application telemetry, cloud inference or automatic report upload. Media copies, transcripts, detections, retained clips and labels stay in `.runtime` until removed. Initial downloads and optional live-stream URLs make network requests. This preview is not a hardened network service. [Privacy](../PRIVACY.md) · [Security](../SECURITY.md)

## License and media

Original project code is licensed under **[MIT](../LICENSE)**. Separately installed dependencies, codecs and model weights retain their own terms. [Third-party notices](../NOTICE.md).

The video and GIF contain separately licensed third-party footage: **Pexels License** for the kissing clip and **CC BY 3.0** for *Tears of Steel*. They are not all MIT or all Creative Commons. [Media credits](../docs/MEDIA_CREDITS.md) and [the media manifest](../media-assets.json) record each creator, source, license, checksum and change. The filter-controls poster is original project artwork under MIT and contains no third-party footage. No endorsement by the creators or performers is implied.

The public package excludes private media, datasets, transcripts, model weights, runtime environments and development Git history. Its only bundled media are the three reviewed assets listed in the media manifest.

[Release notes](../RELEASE_NOTES.md) · [Validation status](../RELEASE_STATUS.md) · [Release process](../RELEASING.md)
