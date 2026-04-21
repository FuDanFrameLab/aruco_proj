"""Tkinter GUI for live ArUco pose preview and synchronized video + CSV recording.

Features:
 - Select + connect a camera. If its name is found in data/camera_intrinsics.json,
   load the intrinsics automatically; otherwise warn and allow manual selection
   of any intrinsics entry in that JSON.
 - Live preview with drawn axes on every detected ArUco marker.
 - Configurable record size; the saved MP4 matches that size.
 - Pose (rvec, tvec) for every known marker ID is recorded per frame to a CSV.
   Missing markers are written as zeros in that frame's row. The CSV includes a
   frame index and a relative timestamp aligned with the video timeline.
"""
from __future__ import annotations

import csv
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
    ARUCO_DICTS,
    DATA_DIR,
    INTRINSICS_JSON,
    get_aruco_dict,
    list_cameras,
    load_json,
    open_camera,
)

RECORD_SIZES = {
    "640x480": (640, 480),
    "960x540": (960, 540),
    "1280x720": (1280, 720),
    "1920x1080": (1920, 1080),
    "2560x1440": (2560, 1440),
    "3840x2160": (3840, 2160),
}

REQUESTED_FPS = {
    "640x480": 60.0,
    "960x540": 60.0,
    "1280x720": 60.0,
    "1920x1080": 30.0,
    "2560x1440": 30.0,
    "3840x2160": 25.0,
}


class RecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ArUco Live Preview & Recorder")

        self.cap: cv2.VideoCapture | None = None
        self.preview_running = False
        self.preview_thread: threading.Thread | None = None

        self.K: np.ndarray | None = None
        self.dist: np.ndarray | None = None
        self.intrinsics_source: str | None = None

        self.recording = False
        self.writer: cv2.VideoWriter | None = None
        self.csv_file = None
        self.csv_writer: csv.writer | None = None
        self.record_start: float | None = None
        self.frame_index = 0
        self.record_size: tuple[int, int] = (1280, 720)
        self.marker_ids_universe: list[int] = []

        self._build_ui()
        self._refresh_cameras()
        self._refresh_intrinsics()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.grid(row=0, column=0, sticky="we")

        ttk.Label(top, text="Camera idx:").grid(row=0, column=0, sticky="w")
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(top, textvariable=self.cam_var, width=6, state="readonly")
        self.cam_combo.grid(row=0, column=1)
        ttk.Button(top, text="Refresh", command=self._refresh_cameras).grid(row=0, column=2)

        ttk.Label(top, text="Camera name:").grid(row=0, column=3, padx=(10, 0))
        self.cam_name_var = tk.StringVar(value="camera0")
        ttk.Entry(top, textvariable=self.cam_name_var, width=14).grid(row=0, column=4)

        ttk.Label(top, text="Dictionary:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.dict_var = tk.StringVar(value="DICT_4X4_50")
        self.dict_combo = ttk.Combobox(
            top, textvariable=self.dict_var, values=list(ARUCO_DICTS.keys()),
            width=20, state="readonly",
        )
        self.dict_combo.grid(row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(top, text="Marker length (m):").grid(row=1, column=3, pady=(6, 0), padx=(10, 0))
        self.marker_len_var = tk.DoubleVar(value=0.025)
        ttk.Entry(top, textvariable=self.marker_len_var, width=8).grid(row=1, column=4, pady=(6, 0))

        ttk.Label(top, text="Intrinsics:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.intr_var = tk.StringVar()
        self.intr_combo = ttk.Combobox(top, textvariable=self.intr_var, width=24, state="readonly")
        self.intr_combo.grid(row=2, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(top, text="Reload", command=self._refresh_intrinsics).grid(row=2, column=3, pady=(6, 0))

        ttk.Label(top, text="Record size:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.size_var = tk.StringVar(value="1280x720")
        ttk.Combobox(
            top, textvariable=self.size_var, values=list(RECORD_SIZES.keys()),
            width=12, state="readonly",
        ).grid(row=3, column=1, sticky="w", pady=(6, 0))

        self.connect_btn = ttk.Button(top, text="Connect", command=self._toggle_connect)
        self.connect_btn.grid(row=4, column=0, pady=8, sticky="w")
        self.record_btn = ttk.Button(top, text="Start Recording", command=self._toggle_record, state="disabled")
        self.record_btn.grid(row=4, column=1, pady=8, sticky="w")

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

    def _refresh_intrinsics(self) -> None:
        data = load_json(INTRINSICS_JSON)
        self.intr_combo["values"] = sorted(data.keys())

    def _load_intrinsics_for(self, name: str) -> bool:
        data = load_json(INTRINSICS_JSON)
        entry = data.get(name)
        if entry is None:
            return False
        self.K = np.array(entry["camera_matrix"], dtype=np.float64)
        self.dist = np.array(entry["dist_coeffs"], dtype=np.float64).reshape(-1)
        self.intrinsics_source = name
        return True

    def _toggle_connect(self) -> None:
        if self.preview_running:
            self._disconnect()
            return
        if not self.cam_var.get():
            messagebox.showwarning("No camera", "No camera selected.")
            return
        idx = int(self.cam_var.get())
        self.cap = open_camera(idx)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Cannot open camera {idx}.")
            self.cap = None
            return

        cam_name = self.cam_name_var.get().strip() or f"camera{idx}"
        w_req, h_req = RECORD_SIZES[self.size_var.get()]
        lookup_key = f"{cam_name}@{w_req}x{h_req}"
        if not self._load_intrinsics_for(lookup_key) and not self._load_intrinsics_for(cam_name):
            self.K = None
            self.dist = None
            self.intrinsics_source = None
            choice = self.intr_var.get().strip()
            if choice:
                self._load_intrinsics_for(choice)
                messagebox.showinfo(
                    "Intrinsics",
                    f"No entry for '{cam_name}'. Using manually selected '{choice}'.",
                )
            else:
                messagebox.showwarning(
                    "No intrinsics",
                    f"No intrinsics for '{cam_name}'. Calibrate the camera, "
                    "or pick one from the Intrinsics dropdown and reconnect.\n"
                    "Preview will continue without pose estimation.",
                )

        self.preview_running = True
        self.connect_btn.configure(text="Disconnect")
        self.record_btn.configure(state="normal")
        self._apply_capture_size()
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()

    def _disconnect(self) -> None:
        if self.recording:
            self._stop_recording()
        self.preview_running = False
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=1.0)
            self.preview_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.connect_btn.configure(text="Connect")
        self.record_btn.configure(state="disabled")

    def _apply_capture_size(self) -> None:
        if self.cap is None:
            return
        key = self.size_var.get()
        w, h = RECORD_SIZES[key]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        fps_req = REQUESTED_FPS.get(key, 30.0)
        self.cap.set(cv2.CAP_PROP_FPS, float(fps_req))

    def _letterbox(self, frame: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        tw, th = target
        h, w = frame.shape[:2]
        if (w, h) == (tw, th):
            return frame
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        out = np.zeros((th, tw, 3), dtype=frame.dtype)
        x = (tw - nw) // 2
        y = (th - nh) // 2
        out[y:y + nh, x:x + nw] = resized
        return out

    def _detect_markers(self, gray: np.ndarray):
        aruco_dict = get_aruco_dict(self.dict_var.get())
        try:
            params = cv2.aruco.DetectorParameters()
        except AttributeError:
            params = cv2.aruco.DetectorParameters_create()
        try:
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        return corners, ids

    def _estimate_poses(self, corners):
        if self.K is None or self.dist is None:
            return None, None
        marker_len = float(self.marker_len_var.get())
        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_len, self.K, self.dist
            )
            return rvecs, tvecs
        except AttributeError:
            half = marker_len / 2.0
            obj = np.array([
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ], dtype=np.float32)
            rvecs, tvecs = [], []
            for c in corners:
                ok, r, t = cv2.solvePnP(obj, c[0], self.K, self.dist)
                if ok:
                    rvecs.append(r.reshape(1, 3))
                    tvecs.append(t.reshape(1, 3))
                else:
                    rvecs.append(np.zeros((1, 3)))
                    tvecs.append(np.zeros((1, 3)))
            return np.array(rvecs), np.array(tvecs)

    def _preview_loop(self) -> None:
        last_t = time.time()
        live_fps = 0.0
        while self.preview_running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            proc_frame = frame

            gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            corners, ids = self._detect_markers(gray)

            rvecs, tvecs = None, None
            if ids is not None and len(ids) > 0:
                rvecs, tvecs = self._estimate_poses(corners)

            vis = proc_frame.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                if rvecs is not None and self.K is not None:
                    axis_len = float(self.marker_len_var.get()) * 0.5
                    for i in range(len(ids)):
                        try:
                            cv2.drawFrameAxes(vis, self.K, self.dist, rvecs[i], tvecs[i], axis_len)
                        except AttributeError:
                            cv2.aruco.drawAxis(vis, self.K, self.dist, rvecs[i], tvecs[i], axis_len)
                        self._draw_pose_label(vis, corners[i], int(ids[i][0]), rvecs[i], tvecs[i])

            if self.recording and self.writer is not None:
                record_frame = self._letterbox(proc_frame, self.record_size)
                self._write_record_row(record_frame, ids, rvecs, tvecs)

            self._show_frame(vis)
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                live_fps = inst if live_fps == 0.0 else live_fps * 0.9 + inst * 0.1
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            msg = (
                f"Intrinsics: {self.intrinsics_source or 'NONE'} | "
                f"Cam: {actual_w}x{actual_h}@{live_fps:4.1f}fps | "
                f"Detected: {0 if ids is None else len(ids)} | "
                f"Recording: {'YES' if self.recording else 'no'} | "
                f"Frames: {self.frame_index}"
            )
            self.status_var.set(msg)

    def _write_record_row(self, frame_bgr, ids, rvecs, tvecs) -> None:
        assert self.writer is not None and self.csv_writer is not None
        self.writer.write(frame_bgr)
        now = time.time()
        t_rel = now - (self.record_start or now)

        detected = {}
        if ids is not None and rvecs is not None and tvecs is not None:
            for i, mid in enumerate(ids.flatten().tolist()):
                detected[int(mid)] = (rvecs[i].reshape(3), tvecs[i].reshape(3))

        row = [self.frame_index, f"{t_rel:.6f}"]
        for mid in self.marker_ids_universe:
            if mid in detected:
                r, t = detected[mid]
                row.extend([1, *r.tolist(), *t.tolist()])
            else:
                row.extend([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.csv_writer.writerow(row)
        self.frame_index += 1

    def _draw_pose_label(self, img, marker_corners, mid: int, rvec, tvec) -> None:
        r = np.asarray(rvec).reshape(3)
        t = np.asarray(tvec).reshape(3)
        pts = np.asarray(marker_corners).reshape(-1, 2)
        x = int(pts[:, 0].min())
        y = int(pts[:, 1].min()) - 6
        lines = [
            f"id={mid}",
            f"t=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]m",
            f"r=[{r[0]:+.3f},{r[1]:+.3f},{r[2]:+.3f}]rad",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        for i, line in enumerate(lines):
            yy = y - (len(lines) - 1 - i) * 14
            if yy < 12:
                yy = int(pts[:, 1].max()) + 14 + i * 14
            cv2.putText(img, line, (x, yy), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(img, line, (x, yy), font, scale, (0, 255, 255), thick, cv2.LINE_AA)

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        max_w = 960
        if w > max_w:
            scale = max_w / w
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        tkimg = ImageTk.PhotoImage(pil)
        self.preview_label.configure(image=tkimg)
        self.preview_label.image = tkimg

    def _toggle_record(self) -> None:
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        size_key = self.size_var.get()
        w, h = RECORD_SIZES[size_key]
        self.record_size = (w, h)

        if self.cap is None:
            return
        self._apply_capture_size()
        ok, probe = self.cap.read()
        if not ok:
            messagebox.showerror("Recorder", "Failed to grab a probe frame.")
            return
        gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
        _, probe_ids = self._detect_markers(gray)
        if probe_ids is None or len(probe_ids) == 0:
            messagebox.showwarning(
                "Recorder",
                "No ArUco markers visible in the current frame. "
                "Make sure all target markers are in view before starting.",
            )
            return
        self.marker_ids_universe = sorted({int(i) for i in probe_ids.flatten().tolist()})

        fps = REQUESTED_FPS.get(size_key, 30.0)
        if self.cap is not None:
            measured = self.cap.get(cv2.CAP_PROP_FPS)
            if measured and measured > 1:
                fps = float(measured)

        os.makedirs(DATA_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cam_name = self.cam_name_var.get().strip() or "camera"
        video_path = os.path.join(DATA_DIR, f"record_{cam_name}_{stamp}.mp4")
        csv_path = os.path.join(DATA_DIR, f"record_{cam_name}_{stamp}.csv")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        if not self.writer.isOpened():
            messagebox.showerror("Recorder", f"Failed to open video writer for {video_path}")
            self.writer = None
            return

        self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        header = ["frame_index", "timestamp_s"]
        for mid in self.marker_ids_universe:
            header.extend([
                f"id{mid}_detected",
                f"id{mid}_rx", f"id{mid}_ry", f"id{mid}_rz",
                f"id{mid}_tx", f"id{mid}_ty", f"id{mid}_tz",
            ])
        self.csv_writer.writerow(header)

        self.record_start = time.time()
        self.frame_index = 0
        self.recording = True
        self.record_btn.configure(text="Stop Recording")

    def _stop_recording(self) -> None:
        self.recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        self.record_btn.configure(text="Start Recording")

    def on_close(self) -> None:
        self._disconnect()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = RecorderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
