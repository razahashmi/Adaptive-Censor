# Demonstration footage, audio, permissions and changes

The opening title carries open-source branding only. **MIT-licensed code** appears on the end card; the software uses the [MIT License](../LICENSE), while the source footage retains its separate terms. A dedicated warning screen at **0:03–0:06**, before any footage, lists adult kissing, a filtered fictional organ prop and original audio containing profanity.

The **63-second introduction** contains one kissing example, a four-panel filter collage, a category-controls illustration, authentic player screenshots, a staged report/correction example, concise information and an audible **original → beep → mute** comparison. Reviewed intervals use the actual project renderer; no recognition model is run when building the video.

## Kissing — Anna Tarazevich / Pexels

- Title: **Close Up Video of People Kissing**, video 5763324.
- Creator and source: [Anna Tarazevich / Pexels](https://www.pexels.com/video/close-up-video-of-people-kissing-5763324/).
- Source download: <https://videos.pexels.com/video-files/5763324/5763324-hd_1920_1080_24fps.mp4>.
- Source SHA256: `f937d4aeb4e02ac15330f5ff3d10c81089844f4d5acc6dbc3f2d197f16530666`.
- Terms: **[Pexels License](https://www.pexels.com/license/)** (`LicenseRef-Pexels`).
- Source excerpt: **00:00–00:06**, played once, with one kiss. Strong blur applies at **4.05–5.65 seconds** within it. The clip's original audio is removed.

The GIF uses this kissing excerpt; the poster contains original promotional graphics only. The Pexels license has its own conditions; the footage is **not MIT, public domain or Creative Commons**. Its restrictions include no implied endorsement, offensive depiction of identifiable people or redistribution as a competing stock library. The source clip is not bundled as a standalone file.

## Filter collage and spoken dialogue — Tears of Steel

**Tears of Steel** (2012) — **(CC) Blender Foundation | [mango.blender.org](https://mango.blender.org/)**. Written and directed by Ian Hubert; produced by Ton Roosendaal.

- Official reuse statement: <https://mango.blender.org/sharing/>.
- License: **[Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/)** (`CC-BY-3.0`).
- Official source: <https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov>.
- Source SHA256: `efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f`.
- Collage source excerpt: **00:57–01:01**, repeated as needed for a six-second simultaneous four-panel comparison. All four panes are filtered throughout: light blur, strong blur, manually selected regional blur and opaque concealment. The exposed brain is a **fictional film prop**, not actual injury. This excerpt is silent.
- Speech source excerpt: **05:57.80–06:00.80**, played three times: original audio, the selected word beeped, then muted. The original film voices are used; no synthetic voice or narration is added.

The spoken line is **“I was young, and a dick.”** The abusive word “dick” is played uncensored in the original example, beeped in the second and muted in the third. Local transcription of the full scene and the isolated clip located it at approximately **360.26–360.54 seconds** in the source. The reviewed censor interval is **2.36–2.90 seconds** within the three-second excerpt, covering the word with a leading/trailing margin. The renderer replaces the full audio mix during this interval; it does not separate the speaker from background audio. These reviewed timings do not establish general speech-recognition accuracy.

The speech excerpt is normalized for listening level. Beeps are the software's actual 880Hz replacement signal. These demonstration beeps are functional filter output, not generated source footage. The original soundtrack is retained only within the speech excerpt; no standalone soundtrack, separate music, website logos or unrelated third-party media are added.

## Included assets and terms

| Asset | Sources | Applicable source terms |
|---|---|---|
| `docs/media/adaptive-censor-demo.mp4` | Kissing stock, film prop and film dialogue | Pexels License for the stock portion; CC BY 3.0 for film portions |
| `docs/media/adaptive-censor-preview.gif` | Anna Tarazevich kissing excerpt | Pexels License |
| `docs/media/adaptive-censor-poster.png` | Original promotional artwork; no source footage | MIT |

At **0:18–0:21**, a six-card category-control illustration follows “Four filters. Your choice.” Its settings are illustrative. The separate thumbnail uses original promotional artwork under MIT.

At **0:21–0:29**, edited authentic screenshots show the real player at blur strengths 20, 40, 100 and opaque concealment. At **0:38–0:46**, a deliberately staged false-match interval is reported with **Wrongly censored**, then excluded in the real timeline. The clean picture returns. These are captured UI states, not continuous screen recording or automatic predictions. A separate local runtime contains only this licensed clip and manual demonstration intervals. Analysis is disabled; its meaningless throughput statistic is cropped out. No private media, reports or user information are shown. Reports in this demonstration stay local and do not train models.

The final eight seconds show the project link prominently, with kissing-footage and film/dialogue attribution in two compact, readable lines below. There are no repeated source-credit footers elsewhere in the video, GIF or poster. Full source terms and modification notices remain on this page. The end card links to **https://github.com/razahashmi/Adaptive-Censor**. [Captions and transcript](DEMO_TRANSCRIPT.md) provide a text version.

Changes: selected excerpts, aspect-preserving resizing, original/filtered comparison, four-panel collage, captions and timeline markings, light/strong/regional blur and concealment, speech normalization and timed mute/beep replacement. Original project code, captions and layout are MIT; that does not relicense underlying footage. Each source retains its own license. The composite is not wholly MIT or wholly Creative Commons. [media-assets.json](../media-assets.json) records each source, asset checksum and size.

No creator or performer endorses the project. The explanatory cards describe supported workflows and clearly marked future home-server work. All examples use reviewed timings; they do not claim automatic visual or speech detection accuracy. Source identities and reuse terms were checked on **18 October 2026**. Raw source files and private recordings remain outside the public package.

## Reproduce

Download the two exact sources above, keep them outside the repository and verify the SHA256 values. With the normal runtime dependencies installed:

```powershell
python make_public_demo.py --source /media/pexels-5763324.mp4 --film-source /media/tears_of_steel_720p.mov --ui-capture /media/reviewed-ui-captures
```

The UI input pack (`adaptive-censor-demo-ui-inputs.zip`, a separate release asset when published) contains seven authentic full-page JPEG screenshots and `CAPTURES.json` with hashes, capture context and the separate MIT/Pexels terms. Extract it outside the repository and pass that folder as `--ui-capture`. You can instead recapture equivalent states using your own local licensed clip, but the manifest must reflect your reviewed screenshots. These inputs are not included in the source archive. Their on-page throughput is explicitly not a benchmark.

The command builds the three reviewed assets and updates `media-assets.json`; it downloads nothing, runs no recognition model and publishes nothing. Optional `--work /path/outside-repository` retains intermediate audio/video and the reviewed timeline for inspection. Check rebuilt assets and refresh the manifest before release: encoded hashes may differ across fonts, platforms and FFmpeg versions.
