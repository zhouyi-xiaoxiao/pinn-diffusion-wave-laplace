"""Redraw the stored accuracy-cost data for the README."""

import html
import json
import math
import os
import subprocess


PROBLEMS = ["heat1d", "laplace2d", "wave1d", "wave2d", "laplace3d"]
LABEL = {
    "heat1d": "Heat",
    "laplace2d": "Lap2",
    "wave1d": "Wave1",
    "wave2d": "Wave2",
    "laplace3d": "Lap3",
}
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def esc(value):
    return html.escape(str(value), quote=True)


def text(x, y, value, size=44, anchor="start", fill=INK, weight="normal"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, sans-serif" '
        f'font-size="{size}px" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{esc(value)}</text>'
    )


def line(x1, y1, x2, y2, stroke=INK2, width=4, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
    )


def circle(x, y, radius, fill, stroke="white", width=2):
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def square(x, y, size, fill, stroke="white", width=2):
    half = size / 2
    return (
        f'<rect x="{x-half:.1f}" y="{y-half:.1f}" width="{size:.1f}" '
        f'height="{size:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def diamond(x, y, size, fill, stroke="white", width=2):
    half = size / 2
    points = (
        f"{x:.1f},{y-half:.1f} {x+half:.1f},{y:.1f} "
        f"{x:.1f},{y+half:.1f} {x-half:.1f},{y:.1f}"
    )
    return f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def log_pos(value, low, high, start, end):
    return start + (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low)) * (end - start)


def superscript(value):
    chars = {
        "-": "⁻",
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }
    return "".join(chars[c] for c in str(value))


def legend_item(parts, x, y, marker, lines):
    marker_x = x + 28
    marker_y = y - 12
    if marker == "fd":
        parts.append(line(marker_x - 27, marker_y, marker_x + 27, marker_y, INK2, 4))
        parts.append(circle(marker_x, marker_y, 9, INK2, "white", 2))
    elif marker == "adam":
        parts.append(circle(marker_x, marker_y, 12, BLUE, "white", 2))
    elif marker == "lbfgs":
        parts.append(square(marker_x, marker_y, 24, ORANGE, "white", 2))
    elif marker == "gpu":
        parts.append(diamond(marker_x, marker_y, 25, INK, "white", 2))
    elif marker == "cpu":
        parts.append(square(marker_x, marker_y, 24, INK, "white", 2))
    elif marker == "load":
        parts.append(square(marker_x, marker_y, 24, "white", INK, 4))
    label_x = x + 84
    for index, label in enumerate(lines):
        parts.append(text(label_x, y + index * 47, label, size=42))


def make_svg(data):
    width, height = 1800, 1900
    left, right, top, bottom = 24, 30, 34, 34
    col_gap, row_gap = 72, 64
    panel_w = (width - left - right - col_gap) / 2
    panel_h = (height - top - bottom - 2 * row_gap) / 3
    x_low, x_high = 1e-4, 2e3
    y_low, y_high = 1e-6, 2.0
    x_grid = list(range(-4, 4))
    y_grid = list(range(-6, 1))
    x_labels = {-4, -2, 0, 2}
    y_labels = {-5, -3, -1}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for index, problem in enumerate(PROBLEMS):
        d = data["problems"][problem]
        row, col = divmod(index, 2)
        px = left + col * (panel_w + col_gap)
        py = top + row * (panel_h + row_gap)
        x0, x1 = px + 196, px + panel_w - 28
        y0, y1 = py + 78, py + panel_h - 132
        parts.append(text(px + 14, py + 53, f"({'abcde'[index]}) {LABEL[problem]}", size=52))

        for exponent in x_grid:
            x = log_pos(10**exponent, x_low, x_high, x0, x1)
            parts.append(line(x, y0, x, y1, GRID, 3))
            parts.append(line(x, y1, x, y1 + 12, INK2, 4))
            if exponent in x_labels:
                parts.append(text(x, y1 + 54, f"10{superscript(exponent)}", size=44, anchor="middle"))

        for exponent in y_grid:
            y = log_pos(10**exponent, y_low, y_high, y1, y0)
            parts.append(line(x0, y, x1, y, GRID, 3))
            parts.append(line(x0 - 12, y, x0, y, INK2, 4))
            if exponent in y_labels:
                parts.append(text(x0 - 20, y + 14, f"10{superscript(exponent)}", size=44, anchor="end"))

        parts.append(line(x0, y1, x1, y1, INK2, 5))
        parts.append(line(x0, y0, x0, y1, INK2, 5))
        parts.append(text((x0 + x1) / 2, y1 + 112, "time of one solve (s)", size=44, anchor="middle"))
        y_label = (y0 + y1) / 2
        parts.append(
            f'<text x="{px + 44:.1f}" y="{y_label:.1f}" transform="rotate(-90 {px + 44:.1f} {y_label:.1f})" '
            f'font-family="DejaVu Sans, sans-serif" font-size="44px" fill="{INK}" text-anchor="middle">relative L² error</text>'
        )

        times = [item["time_s"] for item in d["fd_curve"]]
        errors = [item["rel_l2"] for item in d["fd_curve"]]
        points = " ".join(
            f"{log_pos(t, x_low, x_high, x0, x1):.1f},{log_pos(e, y_low, y_high, y1, y0):.1f}"
            for t, e in zip(times, errors)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{INK2}" stroke-width="5"/>')
        for t, e in zip(times, errors):
            parts.append(circle(log_pos(t, x_low, x_high, x0, x1), log_pos(e, y_low, y_high, y1, y0), 10, INK2, "white", 2))

        first_x = log_pos(times[0], x_low, x_high, x0, x1)
        first_y = log_pos(errors[0], y_low, y_high, y1, y0)
        last_x = log_pos(times[-1], x_low, x_high, x0, x1)
        last_y = log_pos(errors[-1], y_low, y_high, y1, y0)
        parts.append(text(first_x + 18, first_y - 27, f"n = {d['fd_curve'][0]['n']}", size=42, fill=INK2))
        parts.append(text(last_x + 18, last_y - 27, f"n = {d['fd_curve'][-1]['n']}", size=42, fill=INK2))

        for arm, colour, marker in (("adam", BLUE, "circle"), ("adam_lbfgs", ORANGE, "square")):
            geomean = d[arm]["geomean_all"]
            lo, hi = d[arm]["min"], d[arm]["max"]
            t_a = d["time_A_reference_driver_s"][arm]
            t_b = d["time_B_verifier_s"][arm]
            t_c = d["time_C_stacked_gpu_per_network_s"][arm]
            x_start = log_pos(min(t_b["min"], t_c["min"]), x_low, x_high, x0, x1)
            x_a = log_pos(t_a, x_low, x_high, x0, x1)
            y_g = log_pos(geomean, y_low, y_high, y1, y0)
            parts.append(line(x_start, y_g, x_a, y_g, colour, 6))
            x_c = log_pos(0.5 * (t_c["min"] + t_c["max"]), x_low, x_high, x0, x1)
            parts.append(diamond(x_c, y_g, 28, colour, "white", 2))
            x_b = log_pos(t_b["median"], x_low, x_high, x0, x1)
            parts.append(line(x_b, log_pos(lo, y_low, y_high, y1, y0), x_b, log_pos(hi, y_low, y_high, y1, y0), colour, 5))
            if marker == "circle":
                parts.append(circle(x_b, y_g, 14, colour, "white", 3))
                parts.append(circle(x_a, y_g, 14, "white", colour, 4))
            else:
                parts.append(square(x_b, y_g, 28, colour, "white", 3))
                parts.append(square(x_a, y_g, 28, "white", colour, 4))

    legend_px = left + panel_w + col_gap
    legend_py = top + 2 * (panel_h + row_gap)
    legend_x = legend_px + 32
    legend_y = legend_py + 112
    legend_item(parts, legend_x, legend_y, "fd", ["finite differences (n doubled", "along the curve)"])
    legend_item(parts, legend_x, legend_y + 128, "adam", ["PINN, Adam"])
    legend_item(parts, legend_x, legend_y + 192, "lbfgs", ["PINN, Adam → L-BFGS"])
    legend_item(parts, legend_x, legend_y + 256, "gpu", ["GPU batch"])
    legend_item(parts, legend_x, legend_y + 320, "cpu", ["single network, CPU"])
    legend_item(parts, legend_x, legend_y + 384, "load", ["single network, CPU under", "heavy load"])
    parts.append("</svg>")
    return "".join(parts)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "data", "s5_hard_cost.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    svg_path = os.path.join(root, "figures", "s5_hard_cost_readme.svg")
    png_path = os.path.join(root, "figures", "s5_hard_cost_readme.png")
    with open(svg_path, "w", encoding="utf-8") as handle:
        handle.write(make_svg(data))
    subprocess.run(["rsvg-convert", "-o", png_path, svg_path], check=True)
    os.unlink(svg_path)
    print("wrote", png_path)


if __name__ == "__main__":
    main()
