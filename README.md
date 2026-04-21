# ArUco Project

三个相互独立的模块：生成 ChArUco 标定板、相机标定 GUI、ArUco 位姿实时预览与录制 GUI。

## 目录结构

```
aruco_proj/
├── common/
│   └── aruco_utils.py          # 共享：字典、Board 构造、检测、JSON、相机枚举
├── scripts/
│   ├── generate_charuco_board.py   # 功能 1：生成可直接打印的 ChArUco PDF
│   ├── calibration_gui.py          # 功能 2：相机标定 GUI
│   └── recorder_gui.py             # 功能 3：实时预览 + 视频/CSV 录制 GUI
├── data/                       # PDF/PNG、标定板元信息、相机内参、录制结果
├── requirements.txt
└── README.md
```

## 安装

```
pip install -r requirements.txt
```

## 功能 1：生成 ChArUco 标定板 PDF

按 DPI 把标定板像素化，再按毫米放进 PDF，打印时选择 **100%（实际尺寸 / 不缩放）** 即可保证方格尺寸精确。页面留白会被保留，板不会落入不可打印区域。

```
python -m scripts.generate_charuco_board \
    --squares-x 7 --squares-y 5 \
    --square-mm 30 --marker-mm 22 \
    --dictionary DICT_5X5_250 \
    --paper A4 --margin-mm 10 --dpi 600
```

输出：`data/<key>.pdf` + `.png`，并在 `data/charuco_boards.json` 中以板的关键特征（字典/格数/方格/Marker）为 key 记录元信息，供功能 2 直接读取。

## 功能 2：相机标定 GUI

```
python -m scripts.calibration_gui
```

1. 选择相机索引，填写相机名。
2. 从下拉框选择 `charuco_boards.json` 中的板。
3. **Open Camera** 开启实时预览（自动高亮识别到的 ChArUco 角点）。
4. 多角度多姿态下点 **Capture** 采样（≥ 5 张后按钮亮起）。
5. **Calibrate & Save** 完成标定，结果按相机名写入 `data/camera_intrinsics.json`。

## 功能 3：实时位姿预览 & 录制 GUI

```
python -m scripts.recorder_gui
```

- 选相机索引 + 相机名，点 **Connect**：
  - 名字能在 `camera_intrinsics.json` 中匹配 → 自动加载内参。
  - 匹配不到 → 提示缺内参；可在 **Intrinsics** 下拉中手选一套再连接。
- 选择字典与 Marker 实物边长（米），即可看到实时画面及每个 ArUco 码的坐标轴。
- **Record size** 下拉选择录制尺寸；点 **Start Recording**：
  - 视频 → `data/record_<name>_<ts>.mp4`（以所选尺寸保存）。
  - 位姿 → `data/record_<name>_<ts>.csv`，列：`frame_index,timestamp_s`，之后每个 marker ID 固定 7 列 `detected,rx,ry,rz,tx,ty,tz`；未检测到的 ID 该帧写 0，保证按 `frame_index` / `timestamp_s` 能与视频对齐。
