import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

CAPTURE_RE = re.compile(r"^data_capture_(\d+)_(\d+)_(\d+)\.json$")
SENSOR_WIDTH = 32
SENSOR_HEIGHT = 24


def list_pair_folders(root):
    """Subfolders of root that look like a capture session (contain capture JSONs)."""
    root = Path(root)
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and any(CAPTURE_RE.match(f.name) for f in p.iterdir())
    )


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
        return json.load(f)


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
        mask = raw != 0
        buffer[mask] = raw[mask]
        grid = buffer.copy().reshape(SENSOR_HEIGHT, SENSOR_WIDTH)
        sequence.append((data["frame_index"], data["unix_time_ms"], grid))

    return sequence


def compute_normalization_range(grids, low_pct=1, high_pct=99):
    values = np.concatenate([g.ravel() for g in grids])
    vmin, vmax = np.percentile(values, [low_pct, high_pct])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def temps_to_uint8(grid, vmin, vmax):
    scaled = (grid - vmin) / (vmax - vmin) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


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


def find_homography_json(pair_folder):
    pair_folder = Path(pair_folder)
    matches = sorted(pair_folder.glob("*_homography.json"))
    return matches[0] if matches else None


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
