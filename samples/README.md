# Licensed public starter examples

Five short clips (26 seconds total) come from **two already inspected source titles**. They are development examples, not a title-independent test set. Reference labels in `public-suite.json` are **assistant drafts** and need human verification. Unknown categories are not silently counted clean. The organ prop does not stand in for graphic-injury ground truth; the action case is unreviewed.

## Sources and terms

- **Close Up Video of People Kissing**, Anna Tarazevich / [Pexels](https://www.pexels.com/video/close-up-video-of-people-kissing-5763324/), [Pexels License](https://www.pexels.com/license/). Source SHA256 `f937d4aeb4e02ac15330f5ff3d10c81089844f4d5acc6dbc3f2d197f16530666`. Edited, normalized kissing excerpt at 0–8s. Do not imply endorsement or redistribute as a competing stock library.
- **Tears of Steel (2012), (CC) Blender Foundation | [mango.blender.org](https://mango.blender.org/)**, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), [reuse statement](https://mango.blender.org/sharing/). Source SHA256 `efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f`. Edited, normalized excerpts at 27.2–30.2, 57–61, 357.8–360.8 and 465–473s. Contains actual profanity and fictional action/organ imagery. No endorsement is implied.

Keep each source's terms separate. Metadata and original tooling are MIT. No private library, user labels, trained weights or whole source movies are included. This pack is separate from the audited source archive and its showcase assets.

## Reproduce

Download the official source URLs recorded in `public-suite.json`, store them outside the repository with the documented filenames, then:

```powershell
python public_benchmark.py prepare --source-dir /media/sources --output /media/public-starter
python public_benchmark.py run --pack /media/public-starter --mode speech --allow-draft --output /media/diagnostic-speech.json
python public_benchmark.py run --pack /media/public-starter --mode cpu-review --allow-draft --output /media/diagnostic-cpu.json
```

Normal base models and optional small.en must be installed separately. Commands run CPU inference and do not train. Draft scoring requires explicit `--allow-draft`; outputs are prominently diagnostic, not qualification evidence. A human can review the clips and update the **copied pack's** `suite.json` review status/reviewer and exact annotations; document that change rather than relabeling assistant output human automatically. The pack's `FILES.json` binds media content; label edits remain visible in the suite identity/report.

For an independent benchmark, add new titles that have not been used for model selection, obtain human labels, and preserve permanent title assignments. Reference positives alone do not establish pixel coverage. Use the existing human-mask evaluator for independent region annotations.
