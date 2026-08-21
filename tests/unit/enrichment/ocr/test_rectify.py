import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

WORKER_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "docker" / "ocr" / "worker"
sys.path.insert(0, str(WORKER_DIR))

from rectify import rectify_crop, _order_points

def test_order_points():
    # Should be TL, TR, BR, BL
    pts = np.array([
        [10.0, 10.0],  # TL
        [20.0, 10.0],  # TR
        [20.0, 20.0],  # BR
        [10.0, 20.0],  # BL
    ], dtype=np.float32)
    # Shuffle
    shuffled = pts[[2, 0, 3, 1]]
    ordered = _order_points(shuffled)
    assert np.allclose(ordered, pts)

def test_horizontal_rectangle():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:30, 20:80] = 255  # White rectangle

    polygon = [[20.0, 10.0], [80.0, 10.0], [80.0, 30.0], [20.0, 30.0]]
    bbox = [20.0, 10.0, 80.0, 30.0]

    crop, strategy = rectify_crop(img, polygon, bbox)
    assert strategy == "min_area_rect_perspective"
    assert crop.shape[0] == 20
    assert crop.shape[1] == 60
    # Check inner region to avoid boundary interpolation artifacts
    assert np.all(crop[2:18, 2:58] == 255)

def test_45_degree_rotated_rectangle():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # A 45 degree rotated rectangle
    polygon = [
        [50.0, 20.0],
        [80.0, 50.0],
        [50.0, 80.0],
        [20.0, 50.0]
    ]
    bbox = [20.0, 20.0, 80.0, 80.0]
    cv2.fillPoly(img, [np.array(polygon, dtype=np.int32)], (255, 255, 255))

    crop, strategy = rectify_crop(img, polygon, bbox)
    assert strategy == "min_area_rect_perspective"
    # Should be rotated to horizontal, so shape width > height
    # The side length is sqrt(30^2 + 30^2) = ~42.4
    assert crop.shape[0] >= 40 and crop.shape[0] <= 45
    assert crop.shape[1] >= 40 and crop.shape[1] <= 45
    # The center of the crop should be white
    assert np.all(crop[20, 20] == 255)

def test_near_vertical_rectangle():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 40:60] = 255  # Vertical rect (height 60, width 20)
    polygon = [[40.0, 20.0], [60.0, 20.0], [60.0, 80.0], [40.0, 80.0]]
    bbox = [40.0, 20.0, 60.0, 80.0]

    crop, strategy = rectify_crop(img, polygon, bbox)
    assert strategy == "min_area_rect_perspective"
    # It should make the long dimension horizontal
    assert crop.shape[1] == 60 # width is 60
    assert crop.shape[0] == 20 # height is 20
    # Check inner region
    assert np.all(crop[2:18, 2:58] == 255)

def test_partly_out_of_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    polygon = [[-10.0, -10.0], [50.0, -10.0], [50.0, 50.0], [-10.0, 50.0]]
    bbox = [-10.0, -10.0, 50.0, 50.0]
    cv2.rectangle(img, (0, 0), (50, 50), (255, 255, 255), -1)

    crop, strategy = rectify_crop(img, polygon, bbox)
    assert strategy == "min_area_rect_perspective"
    # Because of warpPerspective, the out-of-bounds area will be black (0)
    # Wait, the dimensions should be 60x60
    assert crop.shape[0] == 60
    assert crop.shape[1] == 60
    # The (0,0) of the crop is out of bounds, so it is 0
    assert np.all(crop[5, 5] == 0)
    # The (50,50) of the crop corresponds to (40,40) in image which is white
    assert np.all(crop[40, 40] == 255)

def test_degenerate_polygon_bbox_fallback():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:30, 20:80] = 255
    # only 3 points
    polygon = [[20.0, 10.0], [80.0, 10.0], [80.0, 30.0]]
    bbox = [20.0, 10.0, 80.0, 30.0]

    crop, strategy = rectify_crop(img, polygon, bbox)
    assert strategy == "bbox_fallback"
    assert crop.shape == (20, 60, 3)

def test_all_zero_polygon_invalid():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    polygon = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    bbox = [0.0, 0.0, 0.0, 0.0]

    crop, strategy = rectify_crop(img, polygon, bbox)
    assert strategy == "bbox_fallback"
    assert crop.size == 0

def test_non_finite_polygon():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    polygon = [[float('inf'), 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    bbox = [0.0, 0.0, 10.0, 10.0]

    crop, strategy = rectify_crop(img, polygon, bbox)
    assert strategy == "bbox_fallback"
