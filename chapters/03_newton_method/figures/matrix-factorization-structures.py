#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Use a writable Matplotlib config dir in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle


DEEPBLUE = "#2c3e7c"
SLATE = "#47515c"


def draw_pattern(
    ax: plt.Axes,
    mask: np.ndarray,
    *,
    fill_rgba: tuple[float, float, float, float],
    labels: np.ndarray | None = None,
    label_fontsize: int = 7,
    edge_color: str = SLATE,
    edge_width: float = 0.45,
    border_width: float = 0.9,
    v_splits: list[int] | None = None,
    h_splits: list[int] | None = None,
    split_width: float = 1.1,
) -> None:
    m, n = mask.shape
    for i in range(m):
        for j in range(n):
            fc = fill_rgba if mask[i, j] else (1, 1, 1, 1)
            ax.add_patch(Rectangle((j, m - 1 - i), 1, 1, facecolor=fc, edgecolor=edge_color, linewidth=edge_width))
            if labels is not None and labels[i, j]:
                ax.text(
                    j + 0.5,
                    m - 1 - i + 0.5,
                    labels[i, j],
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                    color="black",
                )

    if v_splits:
        for x in v_splits:
            ax.plot([x, x], [0, m], color=edge_color, lw=split_width)
    if h_splits:
        for y in h_splits:
            ax.plot([0, n], [y, y], color=edge_color, lw=split_width)

    # Outer border (helps the matrices read as objects, not grids).
    ax.add_patch(Rectangle((0, 0), n, m, fill=False, edgecolor=edge_color, linewidth=border_width))

    ax.set_xlim(0, n)
    ax.set_ylim(0, m)
    ax.set_aspect("equal")
    ax.axis("off")


def lower_triangular(n: int) -> np.ndarray:
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return i >= j


def upper_triangular(n: int) -> np.ndarray:
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return i <= j


def block_diagonal(n: int) -> np.ndarray:
    D = np.zeros((n, n), dtype=bool)
    # Mix 1x1 and 2x2 blocks as in the lecture notes.
    blocks: list[int] = []
    remaining = n
    next_block = 1
    while remaining > 0:
        if remaining == 1:
            blocks.append(1)
            break
        if remaining == 2:
            blocks.append(2)
            break
        blocks.append(next_block)
        remaining -= next_block
        next_block = 2 if next_block == 1 else 1
    idx = 0
    for b in blocks:
        D[idx : idx + b, idx : idx + b] = True
        idx += b
    return D


def dense(n: int) -> np.ndarray:
    return np.ones((n, n), dtype=bool)


def diagonal(n: int) -> np.ndarray:
    D = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(D, True)
    return D


def r_trapezoidal(m: int, n: int) -> np.ndarray:
    R = np.zeros((m, n), dtype=bool)
    for i in range(min(m, n)):
        R[i, i:n] = True
    return R


def diagonal_labels(n: int, label: str) -> np.ndarray:
    labels = np.full((n, n), "", dtype=object)
    for i in range(n):
        labels[i, i] = label
    return labels


def sparse_diagonal_labels(n: int, label: str) -> np.ndarray:
    labels = np.full((n, n), "", dtype=object)
    if n > 0:
        labels[0, 0] = label
    if n > 1:
        labels[-1, -1] = label
    return labels


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "matrix-factorization-structures.pdf"
    out_svg = out_dir / "matrix-factorization-structures.svg"

    n = 6
    m_qr, n_qr = 6, 4

    fill = to_rgba(DEEPBLUE, 0.22)

    # Keep labels minimal: show only properties guaranteed by the factorization.
    # - Cholesky: diagonal pivots are positive.
    # - LDL / LU: L is unit lower-triangular.
    chol_L_labels = diagonal_labels(n, r"$+$")
    chol_LT_labels = chol_L_labels.T.copy()
    ldl_L_labels = diagonal_labels(n, "1")
    lu_L_labels = diagonal_labels(n, "1")

    ldl_D_labels = np.full((n, n), "", dtype=object)
    D_mask = block_diagonal(n)
    for i in range(n):
        for j in range(n):
            if D_mask[i, j]:
                # Use lowercase entries matching the matrix name: D -> d_{ij}.
                ldl_D_labels[i, j] = rf"$d_{{{i+1}{j+1}}}$"

    # Determine block boundaries to visually emphasize 1x1 vs 2x2 pivots in D.
    blocks: list[int] = []
    remaining = n
    next_block = 1
    while remaining > 0:
        if remaining == 1:
            blocks.append(1)
            break
        if remaining == 2:
            blocks.append(2)
            break
        blocks.append(next_block)
        remaining -= next_block
        next_block = 2 if next_block == 1 else 1
    D_splits = np.cumsum(blocks).tolist()[:-1]

    spectral_Lambda_labels = np.full((n, n), "", dtype=object)
    for i in range(n):
        spectral_Lambda_labels[i, i] = rf"$\lambda_{i+1}$"

    rows: list[tuple[str, str, tuple[np.ndarray, str, dict], tuple[np.ndarray, str, dict]]] = [
        (
            "Cholesky",
            r"$\mathbf{A}=\mathbf{L}\mathbf{L}^\top$",
            (lower_triangular(n), r"$\mathbf{L}$", {"labels": chol_L_labels, "label_fontsize": 10}),
            (upper_triangular(n), r"$\mathbf{L}^\top$", {"labels": chol_LT_labels, "label_fontsize": 10}),
        ),
        (
            r"LDL$^\top$",
            r"$\mathbf{P}^\top\mathbf{A}\mathbf{P}=\mathbf{L}\mathbf{D}\mathbf{L}^\top$",
            (lower_triangular(n), r"$\mathbf{L}$", {"labels": ldl_L_labels, "label_fontsize": 10}),
            (
                block_diagonal(n),
                r"$\mathbf{D}$",
                {"labels": ldl_D_labels, "label_fontsize": 7, "v_splits": D_splits, "h_splits": D_splits},
            ),
        ),
        (
            "LU",
            r"$\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$",
            (lower_triangular(n), r"$\mathbf{L}$", {"labels": lu_L_labels, "label_fontsize": 10}),
            (upper_triangular(n), r"$\mathbf{U}$", {}),
        ),
        (
            "QR",
            r"$\mathbf{A}\mathbf{P}=\mathbf{Q}\mathbf{R}$",
            (
                dense(m_qr),
                r"$\mathbf{Q}$",
                {"v_splits": [n_qr]},  # show Q = [Q1 Q2]
            ),
            (
                r_trapezoidal(m_qr, n_qr),
                r"$\mathbf{R}$",
                {"h_splits": [m_qr - n_qr]},  # show R = [R1; 0]
            ),
        ),
        (
            "Spectral",
            r"$\mathbf{A}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$",
            (dense(n), r"$\mathbf{Q}$", {}),
            (diagonal(n), r"$\boldsymbol{\Lambda}$", {"labels": spectral_Lambda_labels, "label_fontsize": 8}),
        ),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(7.2, 5.9), constrained_layout=True)
    gs = fig.add_gridspec(len(rows), 3, width_ratios=[1.0, 1.35, 1.35])

    for i, (name, eq, left, right) in enumerate(rows):
        ax_text = fig.add_subplot(gs[i, 0])
        ax_text.axis("off")
        ax_text.text(0.0, 0.72, name, ha="left", va="center", fontsize=11, fontweight="bold")
        ax_text.text(0.0, 0.28, eq, ha="left", va="center", fontsize=9)

        left_mask, left_title, left_opts = left
        right_mask, right_title, right_opts = right

        ax_left = fig.add_subplot(gs[i, 1])
        draw_pattern(ax_left, left_mask, fill_rgba=fill, **left_opts)
        ax_left.set_title(left_title, fontsize=10, pad=2)

        ax_right = fig.add_subplot(gs[i, 2])
        draw_pattern(ax_right, right_mask, fill_rgba=fill, **right_opts)
        ax_right.set_title(right_title, fontsize=10, pad=2)

        if name == "QR":
            # Place block labels inside the matrices (more readable than small
            # per-entry symbols, and closer to the lecture-note sketches).
            ax_left.text(n_qr / 2, m_qr / 2, r"$\mathbf{Q}_1$", ha="center", va="center", fontsize=9)
            ax_left.text(n_qr + (m_qr - n_qr) / 2, m_qr / 2, r"$\mathbf{Q}_2$", ha="center", va="center", fontsize=9)

            x_center = n_qr / 2
            ax_right.text(x_center, (m_qr - n_qr) + n_qr / 2, r"$\mathbf{R}_1$", ha="center", va="center", fontsize=9)
            ax_right.text(x_center, (m_qr - n_qr) / 2, r"$\mathbf{0}$", ha="center", va="center", fontsize=9)

    fig.savefig(out_pdf)
    print(f"Wrote {out_pdf}")
    # Keep text as text in SVG so it stays editable in Inkscape.
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(out_svg)
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
