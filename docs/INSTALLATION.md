# Install and first run

## Windows CPU setup launcher

Download the **Windows setup ZIP** from [Releases](https://github.com/razahashmi/Adaptive-Censor/releases/tag/v0.3.0-preview.1). Extract it and open **Adaptive Censor Setup.exe**. The launcher verifies its adjacent source archive, creates a private Python environment in your chosen folder, installs pinned CPU dependencies, and can download the base models. It needs an existing **64-bit Python 3.13** installation; use the official Python download link in the launcher if it is missing.

The launcher is a CPU setup helper, not a fully bundled offline inference application. Dependencies and models download separately. No administrator rights or firewall changes are required. Signing and independent clean-Windows verification remain pending; see [validation status](../RELEASE_STATUS.md).

Leave **Also download base models** unchecked for the smallest first run. This skips separate model downloads and exercises controls without recognition. Add models later to analyze your own files. The UI's **Check setup** reports missing components.

## Source setup

```powershell
.\Setup.ps1
.\Start.ps1
```

Open `http://127.0.0.1:8502/`; stop with `Stop.ps1`. For a controls-only setup use `Setup.ps1 -SkipModels`. Python 3.13 on Windows is the tested base. The service listens locally and is not a supported household network service.

## First three minutes

1. Run the controls demo to check playback and sound controls.
2. Open a local file, select the analysis passes/models you installed and analyze it.
3. Inspect suggestions, confirm accepted events and correct mistakes.
4. Choose category rules, then compare or export. A completed scan does not guarantee complete recognition.

## Optional configurations

- `Setup.ps1 -Gpu`: optional NVIDIA speech/anatomy support. Use a separate tested environment; avoid overlapping CPU/GPU ONNX packages.
- `Setup.ps1 -SceneVerifier`: separately downloaded NVIDIA scene verifier. This is distinct from the base GPU option.
- `Setup.ps1 -SpeechModel small.en`: larger English speech model.
- Training and segmentation use separate dependencies/environments; see the model card and architecture guide.

## Common problems

| Symptom | Check |
|---|---|
| Scripts are blocked | Use the setup launcher or the manual Python commands below; follow your machine's policy. The project does not disable system protections. |
| Service is missing | Check setup completion and the local `server.log`; confirm another app does not use the chosen port. |
| Scene model unavailable | Install the optional verifier or use review-only suggestions. |
| GPU fails | Check installed/loaded ONNX versions and supported NVIDIA runtime; CPU is the initial supported path. |
| Too many wrong filters | Raise the minimum score, review intervals and report precise counterexamples. Blur strength changes appearance, not recognition. |

For an install report use [the installation issue form](https://github.com/razahashmi/Adaptive-Censor/issues/new?template=installation.yml). Remove paths, names and tokens from logs.

## Manual setup if scripts are blocked

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu-win-py313.lock.txt
.\.venv\Scripts\python.exe launcher.py start
```

This starts the controls demo configuration. To install base models first, run `.\.venv\Scripts\python.exe model_setup.py`. The native setup helper uses the installed Windows .NET Framework 4.x; it bundles no Python or model runtime.
