#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Use a writable Matplotlib config dir in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba


DEEPBLUE = "#2c3e7c"
SLATE = "#47515c"


def draw_grid(ax: plt.Axes, pattern: np.ndarray) -> None:
    m, n = pattern.shape
    fill = to_rgba(DEEPBLUE, 0.35)
    for i in range(m):
        for j in range(n):
            fc = fill if pattern[i, j] else (1, 1, 1, 1)
            ax.add_patch(Rectangle((j, m - 1 - i), 1, 1, facecolor=fc, edgecolor=SLATE, linewidth=0.6))
            # Match the lecture-note style: explicitly show 0/1 entries.
            val = "1" if pattern[i, j] else "0"
            color = "black" if pattern[i, j] else SLATE
            alpha = 0.95 if pattern[i, j] else 0.35
            ax.text(j + 0.5, m - 1 - i + 0.5, val, ha="center", va="center", fontsize=10, color=color, alpha=alpha)
    ax.set_xlim(0, n)
    ax.set_ylim(0, m)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "permutation-matrix.pdf"
    out_svg = out_dir / "permutation-matrix.svg"

    # A 4x4 permutation matching the lecture-note sketch (one 1 per row/column).
    perm = [2, 3, 1, 0]
    n = len(perm)
    P = np.zeros((n, n), dtype=bool)
    for i, j in enumerate(perm):
        P[i, j] = True

    fig, ax = plt.subplots(figsize=(2.2, 2.2), constrained_layout=True)
    draw_grid(ax, P)
    ax.set_title(r"Permutation matrix $\mathbf{P}$", fontsize=10, pad=6)

    fig.savefig(out_pdf)
    # Keep text as text in SVG so it stays editable in Inkscape.
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(out_svg)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
