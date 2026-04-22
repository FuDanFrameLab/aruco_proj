"""Tkinter GUI for calibrating a camera using a previously generated ChArUco board.

Workflow:
 1. Pick a camera index and open it.
 2. Pick a ChArUco board by its key from data/charuco_boards.json.
 3. Preview the live feed; detected corners are highlighted.
 4. Click "Capture" to keep the current view as a calibration sample.
 5. Click "Calibrate" to compute intrinsics and save them to
    data/camera_intrinsics.json keyed by camera name.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.aruco_utils import (  # noqa: E402
    BOARDS_JSON,
    INTRINSICS_JSON,
    CharucoSpec,
    build_charuco_board,
    detect_charuco_corners,
    list_cameras,
    load_json,
    open_camera,
    save_json,
)


class CalibrationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Camera Calibration (ChArUco)")

        self.cap: cv2.VideoCapture | None = None
        self.preview_running = False
        self.preview_thread: threading.Thread | None = None
        self.latest_gray: np.ndarray | None = None
        self.frame_size: tuple[int, int] | None = None

        self.spec: CharucoSpec | None = None
        self.board = None

        self.all_corners: list[np.ndarray] = []
        self.all_ids: list[np.ndarray] = []

        self._build_ui()
        self._refresh_cameras()
        self._refresh_boards()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(top, text="Camera:").grid(row=0, column=0, sticky="w")
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(top, textvariable=self.cam_var, width=10, state="readonly")
        self.cam_combo.grid(row=0, column=1, sticky="w")
        ttk.Button(top, text="Refresh", command=self._refresh_cameras).grid(row=0, column=2)

        ttk.Label(top, text="Name:").grid(row=0, column=3, padx=(10, 0), sticky="w")
        self.cam_name_var = tk.StringVar(value="camera0")
        ttk.Entry(top, textvariable=self.cam_name_var, width=14).grid(row=0, column=4, sticky="w")

        ttk.Label(top, text="Board:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.board_var = tk.StringVar()
        self.board_combo = ttk.Combobox(top, textvariable=self.board_var, width=40, state="readonly")
        self.board_combo.grid(row=1, column=1, columnspan=3, sticky="we", pady=(6, 0))
        ttk.Button(top, text="Reload", command=self._refresh_boards).grid(row=1, column=4, pady=(6, 0))

        ttk.Label(top, text="Resolution:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.res_var = tk.StringVar(value="Native")
        ttk.Combobox(
            top, textvariable=self.res_var,
            values=["Native", "640x480", "960x540", "1024x768", "1280x720", "1920x1080", "2560x1440", "3840x2160"],
            width=12,
        ).grid(row=2, column=1, sticky="w", pady=(6, 0))

        self.open_btn = ttk.Button(top, text="Open Camera", command=self._toggle_camera)
        self.open_btn.grid(row=3, column=0, pady=6, sticky="w")
        self.capture_btn = ttk.Button(top, text="Capture", command=self._capture, state="disabled")
        self.capture_btn.grid(row=3, column=1, pady=6, sticky="w")
        self.clear_btn = ttk.Button(top, text="Clear Samples", command=self._clear_samples)
        self.clear_btn.grid(row=3, column=2, pady=6, sticky="w")
        self.calib_btn = ttk.Button(top, text="Calibrate & Save", command=self._calibrate, state="disabled")
        self.calib_btn.grid(row=3, column=3, pady=6, sticky="w")

        self.preview_label = ttk.Label(self.root)
        self.preview_label.grid(row=1, column=0, padx=6, pady=6)

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w").grid(
            row=2, column=0, sticky="we", padx=6, pady=(0, 6)
        )

    def _refresh_cameras(self) -> None:
        cams = list_cameras()
        self.cam_combo["values"] = [str(i) for i in cams]
        if cams:
            self.cam_combo.current(0)

    def _refresh_boards(self) -> None:
        boards = load_json(BOARDS_JSON)
        keys = sorted(boards.keys())
        self.board_combo["values"] = keys
        if keys:
            self.board_combo.current(0)

    def _apply_capture_size(self) -> None:
        if self.cap is None:
            return
        sel = (self.res_var.get() or "").strip().lower()
        if not sel or sel == "native" or "x" not in sel:
            return
        try:
            w_str, h_str = sel.split("x", 1)
            w, h = int(w_str), int(h_str)
        except ValueError:
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))

    def _toggle_camera(self) -> None:
        if self.preview_running:
            self._stop_camera()
            return
        if not self.cam_var.get():
            messagebox.showwarning("No camera", "No camera selected.")
            return
        if not self.board_var.get():
            messagebox.showwarning("No board", "Select a ChArUco board first.")
            return
        boards = load_json(BOARDS_JSON)
        board_info = boards.get(self.board_var.get())
        if board_info is None:
            messagebox.showerror("Board missing", "Board not found in JSON.")
            return
        self.spec = CharucoSpec.from_dict(board_info)
        self.board = build_charuco_board(self.spec)

        idx = int(self.cam_var.get())
        self.cap = open_camera(idx)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Cannot open camera {idx}.")
            self.cap = None
            return
        self._apply_capture_size()
        self.preview_running = True
        self.open_btn.configure(text="Close Camera")
        self.capture_btn.configure(state="normal")
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()

    def _stop_camera(self) -> None:
        self.preview_running = False
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=1.0)
            self.preview_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.open_btn.configure(text="Open Camera")
        self.capture_btn.configure(state="disabled")

    def _preview_loop(self) -> None:
        while self.preview_running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.latest_gray = gray
            self.frame_size = (gray.shape[1], gray.shape[0])
            ch_corners, ch_ids, corners, ids = detect_charuco_corners(gray, self.board, self.spec)
            vis = frame.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            if ch_corners is not None and ch_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids, (0, 255, 0))
            self._show_frame(vis)
            self.status_var.set(
                f"Samples: {len(self.all_corners)} | "
                f"Current detected charuco corners: "
                f"{0 if ch_corners is None else len(ch_corners)}"
            )

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        max_w = 900
        if w > max_w:
            scale = max_w / w
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        tkimg = ImageTk.PhotoImage(pil)
        self.preview_label.configure(image=tkimg)
        self.preview_label.image = tkimg

    def _capture(self) -> None:
        if self.latest_gray is None or self.board is None or self.spec is None:
            return
        ch_corners, ch_ids, _, _ = detect_charuco_corners(self.latest_gray, self.board, self.spec)
        if ch_corners is None or ch_ids is None or len(ch_corners) < 8:
            messagebox.showwarning("Low quality", "Not enough ChArUco corners detected.")
            return
        self.all_corners.append(ch_corners)
        self.all_ids.append(ch_ids)
        if len(self.all_corners) >= 5:
            self.calib_btn.configure(state="normal")

    def _clear_samples(self) -> None:
        self.all_corners.clear()
        self.all_ids.clear()
        self.calib_btn.configure(state="disabled")

    def _calibrate(self) -> None:
        if self.frame_size is None or self.board is None:
            return
        if len(self.all_corners) < 5:
            messagebox.showwarning("Need more samples", "Capture at least 5 views.")
            return
        try:
            ret, K, dist, _rvecs, _tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=self.all_corners,
                charucoIds=self.all_ids,
                board=self.board,
                imageSize=self.frame_size,
                cameraMatrix=None,
                distCoeffs=None,
            )
        except cv2.error as e:
            messagebox.showerror("Calibration failed", str(e))
            return

        base_name = self.cam_name_var.get().strip() or "camera"
        w, h = self.frame_size
        name = f"{base_name}@{w}x{h}"
        data = load_json(INTRINSICS_JSON)
        data[name] = {
            "camera_index": int(self.cam_var.get()),
            "image_width": self.frame_size[0],
            "image_height": self.frame_size[1],
            "camera_matrix": K.tolist(),
            "dist_coeffs": np.asarray(dist).reshape(-1).tolist(),
            "reprojection_error": float(ret),
            "num_samples": len(self.all_corners),
            "board_key": self.board_var.get(),
        }
        save_json(INTRINSICS_JSON, data)
        messagebox.showinfo(
            "Saved",
            f"Calibration saved for '{name}'.\nReprojection error: {ret:.4f} px",
        )

    def on_close(self) -> None:
        self._stop_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = CalibrationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
