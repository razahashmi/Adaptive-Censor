# Preparing the open-source preview

The public artifact is deliberately separate from the development repository's history, private media, local data inventories and saved runtime. Its only media exception is the three reviewed demo assets with separate MIT graphics, Pexels and CC BY 3.0 source terms in `media-assets.json`. A clean artifact does not certify that making the existing private Git history public is appropriate. GitHub's automatic tag archives contain the tagged repository tree; attaching a cleaned ZIP does not replace those archives ([GitHub release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)).

## Build and verify

```powershell
python build_source_release.py --output dist
python build_source_release.py --verify dist/adaptive-censor-0.3.0-preview.1-source.zip
python release_smoke.py dist/adaptive-censor-0.3.0-preview.1-source.zip --report dist/release-smoke.json
```

`release-files.json` is the explicit publication list. README.md in the artifact uses RELEASE_README.md. The development review-batch configuration is replaced by web/batch.default.json. The source template files are also included so the artifact can rebuild itself. The builder rejects unlisted archive files, links, private runtime/media/model paths, suspicious credential strings and personal absolute paths. Only the exact three `docs/media` asset names are permitted as binary media; they require format/size checks, matching reviewed SHA256 values, each documented source, its own license and attribution/credits. These checks bind the reviewed assets, not arbitrary new footage. The audit is a defense, not a guarantee of legal clearance or recognition of arbitrary secrets.

Create a separate public checkout from the verified artifact instead of copying the development folder:

```powershell
python prepare_public_repository.py dist/adaptive-censor-0.3.0-preview.1-source.zip --destination ../adaptive-censor-public --init-git
```

The destination must not already exist. This copies only verified release files and initializes an empty new Git repository if requested. It copies no old `.git` objects, adds no remote, makes no commit and publishes nothing. Review the prepared tree, then create its first commit. Keep private datasets and the downloaded demo sources outside this checkout.

Review the publication list and the actual ZIP contents before approving publication. Verify the archive and SHA256SUMS independently. Private feedback, labels and source clips must not be added. The existing root MIT license covers original code; keep NOTICE.md and install third-party dependencies/models separately.

The smoke check launches the extracted server with a fresh, isolated runtime, no existing models and offline model settings. It uses the generated demo to test authentication, media access, setup diagnostics, feedback, evidence retention, precise-label ingress, training guards, filtered export, restart persistence and source/evidence deletion. Synthetic labeling fixtures are confined to this disposable runtime. It does not establish fresh dependency installation, real-media accuracy or GPU compatibility on its own. CI installs dependencies in a new Windows environment, runs runtime/browser tests and the archive smoke check. Inspect a successful CI run before claiming those remote gates passed.

## Publication decision

The recommended first public repository starts from the reviewed artifact, with a fresh Git history. Keep the existing development repository private unless its full history has been audited and its owner explicitly approves visibility expansion. The selected destination is https://github.com/razahashmi/Adaptive-Censor. Confirm the final reviewed artifacts before external publication. Publish the source and documents, then tag `v0.3.0-preview.1` and create a **prerelease** with RELEASE_NOTES.md and the verified source ZIP/checksums.

Do not label this release stable or make automatic-safety claims. Keep the existing runtime's 0.3.0 compatibility version separate from the source package's preview suffix. Independent user installation, all-visual-review defaults, hardware and recognition validation remain separate release gates.

## Windows setup and optional media packs

```powershell
python build_windows_release.py dist/adaptive-censor-0.3.0-preview.1-source.zip
python public_benchmark.py bundle --pack /path/to/reviewed/public-starter --output dist/adaptive-censor-public-examples.zip
```

The native Windows helper is original MIT C# compiled with the installed .NET Framework compiler. It verifies its adjacent source ZIP against an embedded checksum before setup. It bundles no Python/model/codec runtime. Distribute the full Windows ZIP with its source ZIP, build receipt, license and checksum. Do not call the executable signed or independently clean-OS tested unless those checks occurred. The setup command can be tested headlessly with `Adaptive Censor Setup.exe --self-test --report setup-check.json`; this does not exercise its GUI or install dependencies.

The licensed starter examples and authentic screenshot input pack are **separate optional release assets**, not additions to the source binary exception. Include their credits and hash manifests. Regenerate `ASSET_SHA256SUMS` after the final artifacts are built. The public reference annotations remain development drafts until a human reviews them; retain unknown categories.
