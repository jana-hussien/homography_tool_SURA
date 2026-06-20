import csv
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CAPTURE_RE = re.compile(r"^data_capture_(\d+)_(\d+)_(\d+)\.json$")
CAM_NAME_RE = re.compile(r"cam(\d+)")
SENSOR_WIDTH = 32
SENSOR_HEIGHT = 24


def discover_camera_ids(pair_folder):
    pair_folder = Path(pair_folder)
    ids = sorted({
        int(m.group(1))
        for f in pair_folder.iterdir()
        if (m := CAPTURE_RE.match(f.name))
    })
    return ids


def detect_pair_cameras(pair_folder):
    ids = discover_camera_ids(pair_folder)
    if len(ids) != 2:
        raise ValueError(
            f"Expected exactly 2 cameras' captures in {pair_folder}, found {len(ids)}: {ids}"
        )
    return ids[0], ids[1]


def list_captures(pair_folder, cam_id):
    pair_folder = Path(pair_folder)
    captures = []
    for f in pair_folder.iterdir():
        m = CAPTURE_RE.match(f.name)
        if m and int(m.group(1)) == cam_id:
            captures.append((int(m.group(2)), f))
    captures.sort(key=lambda item: item[0])
    return [path for _seq, path in captures]


def load_capture(path):
    with open(path) as f:
        text = f.read()
    # The sensor occasionally emits a non-standard "-nan" token (rather than
    # JSON's "NaN") during its first few incomplete frames.
    text = re.sub(r"-nan\b", "NaN", text)
    return json.loads(text)


def build_camera_sequence(pair_folder, cam_id):
    """Returns a list of (frame_index, unix_time_ms, temperature_grid) for one
    camera, in capture order. Any stray zero-valued sensor readings (e.g. the
    sensor's incomplete first frame) are filled in from the previous frame."""
    paths = list_captures(pair_folder, cam_id)
    if not paths:
        raise ValueError(f"No captures found for camera {cam_id} in {pair_folder}")

    sequence = []
    buffer = np.zeros(SENSOR_WIDTH * SENSOR_HEIGHT, dtype=np.float64)
    for path in paths:
        data = load_capture(path)
        raw = np.array(data["data"], dtype=np.float64)
        mask = (raw != 0) & ~np.isnan(raw)
        buffer[mask] = raw[mask]
        # The sensor reads out mirrored relative to the camera's actual
        # field of view; flip horizontally so frames, quad selections, and
        # stitched homographies all agree with what the camera physically sees.
        grid = np.ascontiguousarray(buffer.reshape(SENSOR_HEIGHT, SENSOR_WIDTH)[:, ::-1])
        sequence.append((data["frame_index"], data["unix_time_ms"], grid))

    return sequence


def align_camera_sequences(sequences):
    """Given cam_id -> list of (frame_index, unix_time_ms, grid) (as returned
    by build_camera_sequence), restrict every camera to only the frame_index
    values present in *all* cameras, sorted ascending, so position i in every
    camera's output list is guaranteed to be the same captured moment.

    Without this, zipping sequences by list position silently misaligns
    cameras as soon as any one of them is missing a different set of frames
    than the others (e.g. after deleting corrupt/empty captures one by one
    per camera) — the lists are the same length by luck, not by frame_index.
    """
    grids_by_index = {}
    common_indices = None
    for cam_id, seq in sequences.items():
        grids_by_index[cam_id] = {frame_index: grid for frame_index, _ts, grid in seq}
        indices = set(grids_by_index[cam_id].keys())
        common_indices = indices if common_indices is None else common_indices & indices

    common_indices = sorted(common_indices or [])
    return {
        cam_id: [grids[idx] for idx in common_indices]
        for cam_id, grids in grids_by_index.items()
    }


def compute_normalization_range(grids, low_pct=1, high_pct=99):
    values = np.concatenate([g.ravel() for g in grids])
    vmin, vmax = np.percentile(values, [low_pct, high_pct])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def temps_to_uint8(grid, vmin, vmax):
    scaled = (grid - vmin) / (vmax - vmin) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def heatmap_from_uint8(gray):
    """Colorize a normalized uint8 thermal frame with a JET colormap, keeping
    zero-valued (no-coverage) pixels black rather than "cold blue"."""
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    colored[gray == 0] = (0, 0, 0)
    return colored


def save_homography(pair_folder, pair_name, cam_a_name, cam_b_name, quad_a, quad_b, H):
    pair_folder = Path(pair_folder)
    json_path = pair_folder / f"{pair_name}_homography.json"
    npy_path = pair_folder / f"{pair_name}_H.npy"

    data = {
        "pair_name": pair_name,
        "cam_a": cam_a_name,
        "cam_b": cam_b_name,
        "quad_a": [list(map(float, pt)) for pt in quad_a],
        "quad_b": [list(map(float, pt)) for pt in quad_b],
        "H": H.tolist(),
        "timestamp": datetime.now().isoformat(),
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    np.save(npy_path, H)

    return json_path, npy_path


def load_homography_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_homography_files(root):
    """Recursively find every *_homography.json under root (saved calibration
    pairs may live in their own subfolders) and return the parsed entries.

    Only thermal-to-thermal pairs (cam_a/cam_b both "camN") are returned, since
    this feeds the thermal stitching graph in core.stitcher.build_clusters,
    which keys everything by integer cam id. Other calibration files that may
    live alongside these (e.g. an RGB<->thermal "rgb"/"camN" pair saved by
    ui.rgb_calibration_window) are silently skipped rather than crashing.
    """
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*_homography.json")):
        data = load_homography_json(path)
        match_a = CAM_NAME_RE.match(data["cam_a"])
        match_b = CAM_NAME_RE.match(data["cam_b"])
        if not match_a or not match_b:
            continue
        cam_a = int(match_a.group(1))
        cam_b = int(match_b.group(1))
        H = np.array(data["H"], dtype=np.float64)
        entries.append({"cam_a": cam_a, "cam_b": cam_b, "H": H, "path": path})
    return entries


def discover_capture_session(session_folder):
    """Given a parent session folder (e.g. one containing a thermal-*/
    subfolder with capture JSONs, an rgb-*/ subfolder of images, and a
    sync.csv), auto-detect each piece by content rather than by name so
    the folders can be named anything. Returns
    (capture_folder, rgb_folder, sync_csv_path); the latter two may be
    None if not found. Raises ValueError if no capture JSONs are found
    anywhere under session_folder.
    """
    session_folder = Path(session_folder)

    capture_folder = None
    rgb_folder = None
    for child in sorted(session_folder.iterdir()):
        if not child.is_dir():
            continue
        if discover_camera_ids(child):
            capture_folder = child
        elif any(f.suffix.lower() in (".jpg", ".jpeg", ".png") for f in child.iterdir() if f.is_file()):
            rgb_folder = child

    if capture_folder is None and discover_camera_ids(session_folder):
        capture_folder = session_folder

    if capture_folder is None:
        raise ValueError(f"No thermal capture JSONs found under {session_folder}")

    sync_csv = session_folder / "sync.csv"
    if not sync_csv.exists():
        matches = list(session_folder.rglob("sync.csv"))
        sync_csv = matches[0] if matches else None

    return capture_folder, rgb_folder, sync_csv


def load_sync_table(csv_path):
    """Parse the capture session's sync.csv into a list of row dicts, in
    anchor_index order."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["anchor_index"]))
    return rows


def nearest_rgb_file(sync_rows, frame_idx, column="rgb_files"):
    """Find the rgb filename for frame_idx, falling back to the closest row
    (by index distance) that actually has one if this exact frame has no
    synced RGB capture."""
    if not sync_rows:
        return None
    n = len(sync_rows)
    frame_idx = max(0, min(frame_idx, n - 1))
    for offset in range(n):
        for idx in (frame_idx - offset, frame_idx + offset):
            if 0 <= idx < n:
                name = sync_rows[idx].get(column, "")
                if name:
                    return name
    return None
