"""Shared utilities for the ArUco project.

Provides ArUco dictionary lookup, ChArUco board construction with a consistent
API across OpenCV versions, JSON persistence helpers, and simple camera
enumeration.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BOARDS_JSON = os.path.join(DATA_DIR, "charuco_boards.json")
INTRINSICS_JSON = os.path.join(DATA_DIR, "camera_intrinsics.json")

ARUCO_DICTS: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def load_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict[str, Any]) -> None:
    ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_aruco_dict(name: str) -> Any:
    if name not in ARUCO_DICTS:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[name])


@dataclass
class CharucoSpec:
    """Describes a ChArUco board. All distances in meters."""

    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary: str

    def key(self) -> str:
        return (
            f"{self.dictionary}_{self.squares_x}x{self.squares_y}"
            f"_sq{int(round(self.square_length * 1000))}mm"
            f"_mk{int(round(self.marker_length * 1000))}mm"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_m": self.square_length,
            "marker_length_m": self.marker_length,
            "dictionary": self.dictionary,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CharucoSpec":
        return CharucoSpec(
            squares_x=int(d["squares_x"]),
            squares_y=int(d["squares_y"]),
            square_length=float(d["square_length_m"]),
            marker_length=float(d["marker_length_m"]),
            dictionary=str(d["dictionary"]),
        )


def build_charuco_board(spec: CharucoSpec):
    """Create a CharucoBoard that works on both old and new OpenCV APIs."""
    aruco_dict = get_aruco_dict(spec.dictionary)
    if hasattr(cv2.aruco, "CharucoBoard") and hasattr(cv2.aruco.CharucoBoard, "__init__"):
        try:
            return cv2.aruco.CharucoBoard(
                (spec.squares_x, spec.squares_y),
                spec.square_length,
                spec.marker_length,
                aruco_dict,
            )
        except Exception:
            pass
    return cv2.aruco.CharucoBoard_create(
        spec.squares_x,
        spec.squares_y,
        spec.square_length,
        spec.marker_length,
        aruco_dict,
    )


def draw_charuco_board(board, size_px: tuple[int, int], margin_px: int = 0):
    """Render the board to a grayscale image, compatible across OpenCV versions."""
    try:
        return board.generateImage(size_px, marginSize=margin_px, borderBits=1)
    except AttributeError:
        img = np.zeros((size_px[1], size_px[0]), dtype=np.uint8)
        return board.draw(size_px, img, margin_px, 1)


def detect_charuco_corners(gray: np.ndarray, board, spec: CharucoSpec):
    """Detect ArUco markers and interpolate ChArUco corners."""
    aruco_dict = get_aruco_dict(spec.dictionary)
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()

    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return None, None, corners, ids

    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if retval is None or retval < 1:
        return None, None, corners, ids
    return ch_corners, ch_ids, corners, ids


def list_cameras(max_index: int = 6) -> list[int]:
    """Probe camera indices 0..max_index-1 and return the ones that open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            found.append(i)
            cap.release()
    return found


def open_camera(index: int) -> cv2.VideoCapture:
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    return cap
