# Thermal Camera Calibration Tool

A PyQt5 desktop tool for finding the homography between pairs of cameras in a
multi-camera ultra-low-resolution thermal wearable rig, and for applying those
calibrations to a real, synchronized capture session — stitching the
cameras that have a calibrated neighbor into one or more panoramas, shown
alongside the unprocessed RGB frame.

## Install

```bash
cd thermal_calibration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Expected data layout

**Calibration recordings** (one short hand-wave recording per camera pair,
used only to compute a homography): each pair folder contains the raw
`data_capture_<port>_<frame>_<timestamp>.json` files for exactly two camera
ports, e.g.:

```
thermal_pairs/
  23/
    data_capture_2_0_....json
    data_capture_3_0_....json
    ...
  45/
    data_capture_4_0_....json
    data_capture_5_0_....json
    ...
```

The folder can be named anything — the tool detects which two camera ports
are present from the filenames themselves, not the folder name. Frame N for
one camera is assumed to correspond to frame N for the other camera within
the same pair folder.

**Real capture session** (the actual footage you want to stitch, not
calibration footage): a folder with the capture JSONs for all camera ports
in one place, plus optionally an RGB folder and a `sync.csv` mapping each
thermal frame index to its corresponding RGB filename:

```
latest-capture/
  thermal/
    data_capture_2_0_....json
    data_capture_3_0_....json
    data_capture_4_0_....json
    ...
  rgb/
    rgb_capture_0_0_....jpg
    ...
  sync/
    sync.csv
```

## Usage

```bash
python main.py
```

### 1. Calibrate a camera pair

1. Click "Open Pair Folder" and select a pair folder (e.g. `thermal_pairs/23`).
2. Scrub through frames with the slider, arrow keys, or Play/Pause (10 fps)
   until you find a frame where the hand-wave is visible in both cameras'
   overlapping field of view.
3. If the two cameras loaded as the wrong A/B orientation (e.g. you want the
   physically "later" camera in your rig's chain to be Cam A), click "Swap
   Cam A / Cam B" first — this swaps which camera's frames/labels are A vs
   B and clears any existing quad selections, since they're orientation
   specific.
4. Click "Select Cam A Overlap", then click 4 points on the Cam A image in
   order: top-left, top-right, bottom-right, bottom-left. Repeat with
   "Select Cam B Overlap" on the Cam B image, in the same corner order.
5. Once both quads show "defined", click "Compute Homography". The 3x3
   matrix is shown in the panel (or paste/edit one directly and click "Apply
   Typed Matrix").
6. Toggle "Show Stitched Preview" to see Cam A warped into Cam B's frame and
   blended, and scrub through frames to sanity-check alignment.
7. Click "Save Homography" to write `<camA><camB>_homography.json` and
   `<camA><camB>_H.npy` into the pair folder (e.g. `23_homography.json`).
8. Click "Generate Example Images" to write 5 evenly-spaced sample frames
   (original side-by-side, warped+blended, and stitched-pair panorama) to an
   `examples/` subfolder of the pair folder.

Repeat for every camera pair you want calibrated. Only pairs with real,
overlapping fields of view should be calibrated — chaining a homography
between non-overlapping cameras produces a degenerate, visibly distorted
transform.

### 2. Stitch a real capture session + RGB

Click "Stitch Real Capture + RGB..." to open a separate window. This applies
the homographies you saved in step 1 to the *real* synchronized footage,
rather than to the mismatched calibration recordings themselves:

1. "Select Thermal Capture Folder..." — the folder with the real session's
   capture JSONs for all cameras.
2. "Select Homography Folder..." — a folder searched recursively for every
   `*_homography.json` (e.g. point it at `thermal_pairs/` to pick up all
   saved pairs at once).
3. Cameras are grouped into clusters by following whichever pairwise
   homographies are available — two cameras end up in the same panorama
   only if there's a chain of calibrated pairs connecting them. A camera
   with no calibrated neighbor is shown on its own, unstitched, and the
   status line lists which camera(s) that applies to.
4. Click "Reorder Cameras..." to control which camera is each cluster's
   reference frame — drag a camera to the top of its group. Picking a
   camera in the middle of a chain (rather than at one end) roughly halves
   the number of homographies chained together, which reduces the
   distortion that compounds across a long chain.
5. Optionally select an RGB folder and the session's `sync.csv` to show the
   matching RGB frame alongside the thermal panorama(s), displayed exactly
   as recorded (no processing).
6. Scrub the slider, or use Play/Pause, to step through synchronized frames.
   Each cluster is displayed as a JET heatmap (areas with no camera coverage
   stay black).
7. "Save Current Merged Frame" / "Save All Merged Frames" write the actual
   stitched **float temperature data** (not the heatmap image) as `.npy`
   files to `<capture_folder>/stitched_output/`, named by which cameras are
   in that cluster, e.g. `cam2+cam3+cam5+cam4+cam6_frame_0001.npy`.

## Keyboard shortcuts

- Left / Right arrow: step one frame back / forward (main calibration window)
- Play/Pause button: auto-advance at ~10 fps (both windows)

## File structure

```
thermal_calibration/
  main.py             # MainWindow: pair loading, scrubbing, quad selection, save/export
  ui/
    viewer.py          # ImageViewer (grayscale or heatmapped color, with size capping
                        #   and optional quad click capture) and RGBViewer (raw RGB display)
    controls.py        # ControlPanel: buttons, status labels, H matrix display
    stitch_window.py   # OrderCamerasDialog + CaptureStitchWindow: applies saved
                        #   homographies to a real capture session, clusters cameras by
                        #   connectivity, heatmap preview, float .npy export
  core/
    homography.py       # compute_homography, warp_image, blend_images
    stitcher.py          # build_clusters (groups cameras via available homographies)
                          #   and stitch_frames (dtype-generic: uint8 display or float32 temps)
    io_utils.py           # frame loading, camera/homography discovery, sync.csv parsing,
                           #   JSON/npy save/load
  examples/              # scaffold placeholder (runtime examples are written per-pair)
  requirements.txt
  README.md
```
