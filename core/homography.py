import cv2
import numpy as np


def compute_homography(quad_a, quad_b):
    pts_a = np.array(quad_a, dtype=np.float32)
    pts_b = np.array(quad_b, dtype=np.float32)
    H, _mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC)
    if H is None:
        raise ValueError("Homography computation failed; check that points are not collinear")
    return H


def warp_image(img, H, output_size):
    """output_size is (width, height), matching cv2's dsize convention."""
    return cv2.warpPerspective(
        img, H, output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def blend_images(img_a_warped, img_b, alpha=0.5):
    img_a = img_a_warped.astype(np.float32)
    img_b = img_b.astype(np.float32)
    blended = alpha * img_a + (1 - alpha) * img_b
    return np.clip(blended, 0, 255).astype(np.uint8)


def _signed_quad_area(quad):
    """Shoelace formula. Sign indicates rotational direction (clockwise vs
    counter-clockwise) of the 4 points, in image pixel coordinates (y grows downward)."""
    pts = np.array(quad, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4))


def check_consistent_handedness(quad_a, quad_b):
    """A clean, non-crossed quad on EACH image individually does NOT guarantee the
    mapping between them isn't mirrored -- that only shows up as a sign flip in
    rotational direction (clockwise on one image, counter-clockwise on the other),
    which is invisible if you only look at one image's points in isolation. Returns
    True if consistent (likely fine), False if a mirror is likely present."""
    area_a, area_b = _signed_quad_area(quad_a), _signed_quad_area(quad_b)
    return (area_a > 0) == (area_b > 0)
