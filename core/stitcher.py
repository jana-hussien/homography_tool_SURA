import numpy as np

from core.homography import warp_image


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


def build_clusters(homography_entries, cam_ids):
    """Group cam_ids into connected clusters using whatever pairwise
    homographies are available (cameras with no calibrated neighbor end up
    in their own size-1 cluster). Each entry's H maps cam_a's coordinates
    into cam_b's coordinates. Returns a list of dicts: {"cams": [...],
    "transforms": [...]} where transforms[i] maps cams[i] into the
    cluster's reference frame (cams[0]).

    Each cluster's reference frame is the first camera in cam_ids (in the
    given order) that belongs to it — callers that want a specific camera
    to be the reference should put it first in cam_ids (see
    OrderCamerasDialog), since chaining fewer hops from the reference
    compounds less per-pair calibration error."""
    graph = {cam: [] for cam in cam_ids}
    for entry in homography_entries:
        a, b, H = entry["cam_a"], entry["cam_b"], entry["H"]
        if a not in graph or b not in graph:
            continue
        graph[a].append((b, np.linalg.inv(H)))  # maps b -> a
        graph[b].append((a, H))                  # maps a -> b

    visited = set()
    clusters = []
    for root_cam in cam_ids:
        if root_cam in visited:
            continue
        visited.add(root_cam)
        cams = [root_cam]
        transforms = [np.eye(3)]
        transform_to_root = {root_cam: np.eye(3)}
        queue = [root_cam]
        while queue:
            current = queue.pop(0)
            for neighbor, T_neighbor_to_current in graph[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                transform_to_root[neighbor] = transform_to_root[current] @ T_neighbor_to_current
                cams.append(neighbor)
                transforms.append(transform_to_root[neighbor])
                queue.append(neighbor)
        clusters.append({"cams": cams, "transforms": transforms})
    return clusters


def stitch_frames(frames, transforms):
    """
    frames: list of single-channel np arrays (uint8 display frames, or
    float32 raw temperature grids), one per camera, same sync index.
    transforms: list of 3x3 homographies mapping each camera into the cam1 frame.
    Returns the stitched panorama in the same dtype as the input frames,
    zero-filled where no camera has coverage.
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

    canvas = np.zeros((canvas_h, canvas_w), dtype=frames[0].dtype)

    for frame, T in zip(frames, transforms):
        full_T = translation @ T
        warped = warp_image(frame, full_T, (canvas_w, canvas_h))
        mask = warped > 0
        canvas[mask] = warped[mask]

    return canvas
