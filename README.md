# cc2fr face recognition prototype

This repository contains a cross-platform (macOS + Windows) openFrameworks app in `facerec/` that performs:

- face detection (YuNet)
- face recognition (SFace embeddings vs gallery)
- image, video, and webcam input modes

The workflow is:

1. bootstrap dependencies and assets (`scripts/bootstrap.py`)
2. build the app (`scripts/build.py`)
3. run the app or run diagnostics

## Quickstart Commands

Run from the repository root.

### macOS

```bash
python3 scripts/bootstrap.py
python3 scripts/build.py
python3 scripts/build.py --run
```

Self diagnostics:

```bash
python3 scripts/build.py --check
```

### Windows (MSYS2 MinGW64 shell)

```bash
python3 scripts/bootstrap.py
python3 scripts/build.py
python3 scripts/build.py --run
```

Self diagnostics:

```bash
python3 scripts/build.py --check
```

---

## 1) First checkout setup

Run these commands from the repository root.

### Prerequisites

- `python3`
- `git`
- `tar` (macOS default)

Platform toolchains:

- macOS: Xcode + command line tools
- Windows: MSYS2 MinGW64 shell with `make` available

### Bootstrap

```bash
python3 scripts/bootstrap.py
```

What bootstrap does (idempotent, safe to rerun):

- installs openFrameworks 0.12.1 into `of/` if missing
- clones `ofxCv` into `of/addons/ofxCv`
- verifies bundled OpenCV is >= 4.5.4
- downloads YuNet + SFace ONNX models into `facerec/bin/data/models/`
- downloads sample media into `facerec/bin/data/samples/`
- downloads example recognition gallery into `facerec/bin/data/gallery/`
- runs openFrameworks projectGenerator for your platform
- on macOS, sets camera permission text in `facerec/openFrameworks-Info.plist`

If bootstrap succeeds, it prints:

```text
[bootstrap] done. Next: python3 scripts/build.py
```

---

## 2) Compile

### macOS (Xcode build via script)

```bash
python3 scripts/build.py
```

### Windows (from MSYS2 MinGW64 shell)

```bash
python3 scripts/build.py
```

### Optional build modes

Debug build:

```bash
python3 scripts/build.py --debug
```

Build + immediate self-test:

```bash
python3 scripts/build.py --check
```

---

## 3) Run the app

Build and launch in one step:

```bash
python3 scripts/build.py --run
```

Or run the produced binary directly:

- macOS app bundle output is in `facerec/bin/` (for example `facerecRelease.app` or `facerec.app`)
- Windows executable output is in `facerec/bin/facerec.exe` (or `facerec_debug.exe` for debug)

Inside the app:

- open image/video from the GUI buttons or drag and drop
- toggle webcam mode in the panel
- tune detection score threshold and recognition match threshold
- load a custom gallery from the panel

---

## 4) Self diagnostics

Diagnostics run from the built binary and are intended for CI/headless checks as well as local verification.

### A) Dependency/model self-test

From the repo root:

```bash
python3 scripts/build.py --check
```

Equivalent direct binary usage:

```bash
# run the built executable with:
--selftest
```

What it verifies:

- openFrameworks and OpenCV load correctly
- OpenCV version is >= 4.5.4
- YuNet and SFace model files exist in data/models
- YuNet detector and SFace recognizer can be instantiated

Exit code:

- `0` = pass
- non-zero = fail

### B) Headless detection on one image

Run the binary with:

```bash
--detect <image>
```

Example image paths after bootstrap:

- `samples/messi5.jpg`
- `samples/group.pgm`

Output includes:

- total face count
- per-face box (`x y w h`) and confidence

### C) Headless recognition on one image

Run the binary with:

```bash
--identify <image>
```

Example image paths after bootstrap:

- `samples/messi-worldcup.jpg`
- `samples/ronaldo-worldcup.jpg`

Output includes:

- gallery person count
- face count
- per-face recognized name (or `unknown`), best match, and score

---

## 5) Troubleshooting

- If build files are missing, rerun:
  - `python3 scripts/bootstrap.py`
- If models/gallery are missing or corrupted, rerun:
  - `python3 scripts/bootstrap.py`
- If webcam does not open on macOS, check system camera permission for the built app.
- If Windows build says `make` is missing, run from an MSYS2 MinGW64 shell.

---

## 6) Useful paths

- app source: `facerec/src/`
- app data (models, gallery, samples): `facerec/bin/data/`
- bootstrap script: `scripts/bootstrap.py`
- build script: `scripts/build.py`
