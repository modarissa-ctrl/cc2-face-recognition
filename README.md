# cc2fr face recognition prototype

This repository contains a cross-platform (macOS + Windows) openFrameworks app in `facerec/` that performs:

- face detection (YuNet)
- face recognition (SFace embeddings vs gallery)
- image, video, and webcam input modes (with a multi-webcam selector)
- face tracking (own IoU/centroid tracker: stable per-face IDs on video/webcam)
- liveness detection (own blink + mouth-movement detection: LIVE / PHOTO? flag per tracked face)

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
- toggle webcam mode in the panel; if more than one camera is attached,
  pick which one with the webcam device selector (it shows the current
  device name, e.g. `webcam 1/2: FaceTime HD Camera`), cycle with the `[`
  and `]` keys, and hit `Refresh webcams` to rescan after plugging one in
- toggle face tracking — each face on video/webcam gets a stable `#id` label
- toggle liveness — tracked faces earn a `LIVE` flag by blinking or moving
  their mouth (yawn, talking), or a `PHOTO?` flag after ~7 s without either
  (requires tracking to be on; nothing is shown while the verdict is pending)
- tune detection score threshold and recognition match threshold
- load a custom gallery from the panel
- enroll a face into the gallery at runtime: with tracking on and a face
  visible (it shows a `#id` label), press the `A` key or click
  `add face by id...`, enter the face's `#id`, then type the person's name.
  The cropped face is saved under `gallery/<name>/` and is picked up the
  next time the app starts (the running session is not reloaded)

### Liveness limitations

Liveness is a demo of the blink/mouth-movement technique, not anti-spoofing:
a replayed video of a blinking person passes, and a live person who stares
motionlessly gets flagged `PHOTO?`. The signals are measured photometrically
(dark-pixel fraction around the eyes — referenced to the face's overall
brightness — and between the mouth corners), so they need a reasonably lit,
reasonably sized face; heavy glasses glare or strong shadows can hide blinks.

Blinks are detected by *prominence*: the openness signal drifts with distance
and lighting, but a blink dips it by roughly a fixed amount below the recent
eyes-open level, so the detector keys on that drop (from a peak-held open
reference) rather than a fixed threshold, with a stable-open arming gate and
closure-duration bounds to reject camera/photo noise. Because the photometric
signal is shallow and noisy on real faces, it will miss some blinks and the
live/photo signals genuinely overlap — so once a face has proven live (any
blink or mouth movement) its `LIVE` flag is held through longer quiet
stretches, while a face that has *never* moved still flips to `PHOTO?`.

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
- face tracker logic on synthetic detections (stable IDs, position-based matching, dropout tolerance)
- blink and mouth-movement state machines on synthetic signals (dips/rises, noise and static-feature rejection)
- photometric eye-openness and mouth-darkness measures on drawn patches
- end-to-end liveness verdicts (LIVE for a blinking or yawning face, PHOTO? for a motionless one) on rendered frames with a fake clock

Exit code:

- `0` = pass
- non-zero = fail

### A2) Headless liveness replay on a video

Run the binary with:

```bash
--liveness-replay <video> [fps]
```

Decodes a video with OpenCV and runs the real detect → track → liveness
pipeline frame by frame (sub-sampling to ~`fps`, default 23, to mirror the
app's detection rate), printing per-frame eye/mouth signals, blink/mouth
events, and the verdict, plus a summary of total blinks/movements and
LIVE/PHOTO?/PENDING frame counts. This is the tool used to tune the blink
thresholds against real footage in `test/videos/`.

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
