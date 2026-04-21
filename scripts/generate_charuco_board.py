"""Generate a ChArUco calibration board sized for direct, non-scaled PDF printing.

The board is rasterized at the printer DPI so the physical square size in
millimeters is preserved when the PDF is printed at 100% scale. Page margins
are honored so the board never falls into the printer's unprintable area.

Usage:
    python -m scripts.generate_charuco_board \
        --squares-x 7 --squares-y 5 \
        --square-mm 30 --marker-mm 22 \
        --dictionary DICT_5X5_250 \
        --paper A4 --margin-mm 10 --dpi 600
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
from reportlab.lib.pagesizes import A3, A4, LETTER
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.aruco_utils import (  # noqa: E402
    BOARDS_JSON,
    CharucoSpec,
    build_charuco_board,
    draw_charuco_board,
    ensure_data_dir,
    load_json,
    save_json,
)

PAPER_SIZES = {"A4": A4, "A3": A3, "LETTER": LETTER}


def mm_to_px(mm_value: float, dpi: int) -> int:
    return int(round(mm_value / 25.4 * dpi))


def generate(
    squares_x: int,
    squares_y: int,
    square_mm: float,
    marker_mm: float,
    dictionary: str,
    paper: str,
    margin_mm: float,
    dpi: int,
    output_pdf: str,
    output_png: str | None = None,
) -> None:
    if marker_mm >= square_mm:
        raise ValueError("marker_mm must be strictly smaller than square_mm")

    page_w_pt, page_h_pt = PAPER_SIZES[paper]
    page_w_mm = page_w_pt / mm
    page_h_mm = page_h_pt / mm

    board_w_mm = squares_x * square_mm
    board_h_mm = squares_y * square_mm

    def _fits(pw: float, ph: float) -> bool:
        return board_w_mm <= pw - 2 * margin_mm and board_h_mm <= ph - 2 * margin_mm

    if _fits(page_w_mm, page_h_mm):
        landscape = False
    elif _fits(page_h_mm, page_w_mm):
        landscape = True
        page_w_mm, page_h_mm = page_h_mm, page_w_mm
    else:
        raise ValueError(
            f"Board {board_w_mm:.1f}x{board_h_mm:.1f}mm does not fit inside "
            f"{paper} (portrait or landscape) with {margin_mm}mm margin. "
            "Reduce squares, square size, or margin, or use a larger paper."
        )
    pagesize = (page_w_mm * mm, page_h_mm * mm)

    spec = CharucoSpec(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_mm / 1000.0,
        marker_length=marker_mm / 1000.0,
        dictionary=dictionary,
    )
    board = build_charuco_board(spec)

    img_w_px = mm_to_px(board_w_mm, dpi)
    img_h_px = mm_to_px(board_h_mm, dpi)
    img = draw_charuco_board(board, (img_w_px, img_h_px), margin_px=0)

    ensure_data_dir()
    if output_png is None:
        output_png = os.path.splitext(output_pdf)[0] + ".png"
    cv2.imwrite(output_png, img)

    c = canvas.Canvas(output_pdf, pagesize=pagesize)
    x_mm = (page_w_mm - board_w_mm) / 2.0
    y_mm = (page_h_mm - board_h_mm) / 2.0
    c.drawImage(
        output_png,
        x_mm * mm,
        y_mm * mm,
        width=board_w_mm * mm,
        height=board_h_mm * mm,
        preserveAspectRatio=False,
        anchor="sw",
    )
    c.setFont("Helvetica", 8)
    orient = "landscape" if landscape else "portrait"
    label = (
        f"{spec.key()}  |  {squares_x}x{squares_y}  "
        f"square={square_mm}mm marker={marker_mm}mm  |  "
        f"paper={paper}({orient}) margin={margin_mm}mm dpi={dpi}  |  "
        "PRINT AT 100% (no scaling)"
    )
    c.drawString(margin_mm * mm, (margin_mm / 2.0) * mm, label)
    c.showPage()
    c.save()

    boards = load_json(BOARDS_JSON)
    boards[spec.key()] = {
        **spec.to_dict(),
        "paper": paper,
        "margin_mm": margin_mm,
        "dpi": dpi,
        "board_width_mm": board_w_mm,
        "board_height_mm": board_h_mm,
        "pdf_path": os.path.abspath(output_pdf),
        "png_path": os.path.abspath(output_png),
    }
    save_json(BOARDS_JSON, boards)
    print(f"Saved PDF:  {output_pdf}")
    print(f"Saved PNG:  {output_png}")
    print(f"Registered board key '{spec.key()}' in {BOARDS_JSON}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a printable ChArUco board PDF.")
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=5)
    p.add_argument("--square-mm", type=float, default=30.0)
    p.add_argument("--marker-mm", type=float, default=22.0)
    p.add_argument("--dictionary", default="DICT_5X5_250")
    p.add_argument("--paper", choices=list(PAPER_SIZES.keys()), default="A4")
    p.add_argument("--margin-mm", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--output", default=None, help="Output PDF path (default: data/<key>.pdf)")
    args = p.parse_args()

    spec = CharucoSpec(
        args.squares_x,
        args.squares_y,
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        args.dictionary,
    )
    ensure_data_dir()
    output_pdf = args.output or os.path.join(
        os.path.dirname(BOARDS_JSON), f"{spec.key()}.pdf"
    )
    generate(
        args.squares_x,
        args.squares_y,
        args.square_mm,
        args.marker_mm,
        args.dictionary,
        args.paper,
        args.margin_mm,
        args.dpi,
        output_pdf,
    )


if __name__ == "__main__":
    main()
