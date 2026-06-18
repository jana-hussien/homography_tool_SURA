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
