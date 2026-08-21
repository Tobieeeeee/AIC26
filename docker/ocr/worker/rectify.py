import cv2
import numpy as np
import math
from typing import List, Tuple

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # To handle 45-degree ambiguous cases for sum/diff method:
    # A more robust way:
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # For right most, the top point has the smaller y
    # But if the box is heavily rotated, we can use distance to tl.
    # Actually just sorting by Y is standard and works for minAreaRect points.
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _bbox_fallback(image: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, str]:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Clip to image boundaries
    x1_c = max(0, int(round(x1)))
    y1_c = max(0, int(round(y1)))
    x2_c = min(w, int(round(x2)))
    y2_c = min(h, int(round(y2)))

    if x2_c <= x1_c or y2_c <= y1_c:
        return np.array([]), "bbox_fallback"

    crop = image[y1_c:y2_c, x1_c:x2_c]
    return crop, "bbox_fallback"

def rectify_crop(image: np.ndarray, polygon: List[List[float]], bbox: List[float]) -> Tuple[np.ndarray, str]:
    # Check fallback conditions
    if len(polygon) < 4:
        return _bbox_fallback(image, bbox)

    for p in polygon:
        if not (math.isfinite(p[0]) and math.isfinite(p[1])):
            return _bbox_fallback(image, bbox)

    pts = np.array(polygon, dtype=np.float32)

    # Compute minAreaRect
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = _order_points(box)

    (tl, tr, br, bl) = box

    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(height_left), int(height_right))

    if max_width < 2 or max_height < 2:
        return _bbox_fallback(image, bbox)

    area = max_width * max_height
    if area <= 0:
        return _bbox_fallback(image, bbox)

    # Make long text-line dimension horizontal
    if max_height > max_width:
        # Vertical: rotate 90 degrees clockwise
        dst = np.array([
            [max_height, 0],
            [max_height, max_width],
            [0, max_width],
            [0, 0]
        ], dtype=np.float32)
        target_w, target_h = max_height, max_width
    else:
        # Horizontal
        dst = np.array([
            [0, 0],
            [max_width, 0],
            [max_width, max_height],
            [0, max_height]
        ], dtype=np.float32)
        target_w, target_h = max_width, max_height

    try:
        M = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h))
        return warped, "min_area_rect_perspective"
    except Exception:
        return _bbox_fallback(image, bbox)
