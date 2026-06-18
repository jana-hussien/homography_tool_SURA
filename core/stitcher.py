import numpy as np

from core.homography import warp_image
from core.io_utils import find_homography_json, load_homography_json


def load_chain_homographies(pair_folders_in_order):
    """Each folder must contain a saved homography JSON. The list must be in
    camera order (pair_1_2, pair_2_3, ...). Returns H_i mapping cam_i -> cam_(i+1)."""
    homographies = []
    for folder in pair_folders_in_order:
        json_path = find_homography_json(folder)
        if json_path is None:
            raise FileNotFoundError(f"No homography JSON found in {folder}")
        data = load_homography_json(json_path)
        H = np.array(data["H"], dtype=np.float64)
        homographies.append(H)
    return homographies


def build_reference_transforms(homographies):
    """
    homographies[i] maps cam_(i+1) coordinates -> cam_(i+2) coordinates
    (homographies[0] = H_12 maps cam1 -> cam2, etc.)
    Returns transforms[i] mapping camera (i+1) into the cam1 reference frame.
    """
    transforms = [np.eye(3)]
    for H in homographies:
        H_inv = np.linalg.inv(H)
        transforms.append(transforms[-1] @ H_inv)
    return transforms


def _transform_corners(width, height, T):
    corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float64)
    corners_h = np.hstack([corners, np.ones((4, 1))])
    transformed = (T @ corners_h.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def compute_canvas_bounds(frame_shapes, transforms):
    all_points = []
    for (h, w), T in zip(frame_shapes, transforms):
        all_points.append(_transform_corners(w, h, T))
    all_points = np.vstack(all_points)
    min_x, min_y = np.floor(all_points.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_points.max(axis=0)).astype(int)
    return min_x, min_y, max_x, max_y


def stitch_frames(frames, transforms):
    """
    frames: list of grayscale np arrays, one per camera, same sync index.
    transforms: list of 3x3 homographies mapping each camera into the cam1 frame.
    Returns the stitched panorama as a uint8 grayscale image, black-filled where
    no camera has coverage.
    """
    shapes = [f.shape for f in frames]
    min_x, min_y, max_x, max_y = compute_canvas_bounds(shapes, transforms)
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1],
    ], dtype=np.float64)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for frame, T in zip(frames, transforms):
        full_T = translation @ T
        warped = warp_image(frame, full_T, (canvas_w, canvas_h))
        mask = warped > 0
        canvas[mask] = warped[mask]

    return canvas
