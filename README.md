# Thermal Camera Calibration Tool

A PyQt5 desktop tool for finding the homography between adjacent cameras in a
6-camera ultra-low-resolution thermal wearable rig, and for stitching all six
views into a single panorama.

## Install

```bash
cd thermal_calibration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Expected data layout

```
data/
  pair_1_2/
    cam1/
      frame_0001.png
      frame_0002.png
      ...
    cam2/
      frame_0001.png
      ...
  pair_2_3/
    cam2/
    cam3/
  ...
```

Each pair folder must contain exactly two camera subfolders. Frame N in one
camera's folder is assumed to correspond to frame N in the other camera's
folder within the same pair.

## Usage

### 1. Calibrate a single pair

```bash
python main.py
```

1. Select a `pair_X_Y/` folder when prompted (or via "Open Pair Folder").
2. Scrub through frames with the slider, arrow keys, or Play/Pause (10 fps)
   until you find a frame where the hand-wave is visible in both cameras'
   overlapping field of view.
3. Click "Select Cam A Overlap", then click 4 points on the Cam A image in
   order: top-left, top-right, bottom-right, bottom-left. Repeat with
   "Select Cam B Overlap" on the Cam B image, in the same corner order.
4. Once both quads show "defined", click "Compute Homography". The 3x3
   matrix is shown in the panel.
5. Toggle "Show Warped Preview" to see Cam A warped into Cam B's frame and
   alpha-blended, and scrub through frames to sanity-check alignment.
6. Click "Save Homography" to write `pair_X_Y_homography.json` and
   `pair_X_Y_H.npy` into the pair folder.
7. Click "Generate Example Images" to write 5 evenly-spaced sample frames
   (original side-by-side, warped+blended, and stitched-pair panorama) to an
   `examples/` subfolder of the pair folder.

Repeat for every adjacent pair (`pair_1_2`, `pair_2_3`, ..., `pair_5_6`).

### 2. Stitch all six cameras

1. Click "Stitch All Pairs..." in the main window.
2. Select the root folder that contains all `pair_*` subfolders (each must
   already have a saved homography JSON from step 1).
3. Scrub the slider to choose a synchronized frame index; the full 6-camera
   panorama preview updates live, using Cam 1 as the reference frame and
   chaining homographies across all pairs.
4. Use "Save Current Panorama" or "Save All Frames" to write PNGs to a
   `stitched_output/` folder under the root folder.

## Keyboard shortcuts

- Left / Right arrow: step one frame back / forward
- Play/Pause button: auto-advance at ~10 fps

## File structure

```
thermal_calibration/
  main.py             # MainWindow: pair loading, scrubbing, quad selection, save/export
  ui/
    viewer.py          # ImageViewer: upscaled display + quad click capture/overlay
    controls.py        # ControlPanel: buttons, status labels, H matrix display
    stitch_window.py   # StitchWindow: multi-pair homography chaining + panorama preview
  core/
    homography.py       # compute_homography, warp_image, blend_images
    stitcher.py          # homography chaining and multi-camera panorama stitching
    io_utils.py          # frame loading, folder detection, JSON/npy save/load
  examples/              # scaffold placeholder (runtime examples are written per-pair)
  requirements.txt
  README.md
```
