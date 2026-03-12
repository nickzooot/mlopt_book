#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE_DIR = Path("/tmp/matplotlib-cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


INDIGO = "#5b3c8c"
TEAL = "#168f89"
CRIMSON = "#c23b22"
SLATE = "#47515c"


plt.rcParams.update(
    {
        "figure.figsize": (6.0, 3.15),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.labelcolor": SLATE,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.family": "serif",
        "font.size": 9,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{amsmath,amssymb}",
                r"\usepackage{physics}",
                r"\usepackage{bm}",
            ]
        ),
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    }
)


def rotation(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def quadratic_gap(A: np.ndarray, x_star: np.ndarray, x: np.ndarray) -> float:
    e = x - x_star
    return 0.5 * float(e @ (A @ e))


def exact_line_search(A: np.ndarray, x_star: np.ndarray, x: np.ndarray, d: np.ndarray) -> float:
    g = A @ (x - x_star)
    return -float(g @ d) / float(d @ (A @ d))


def linear_cg_two_steps(A: np.ndarray, x_star: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g0 = A @ (x0 - x_star)
    d0 = -g0
    alpha0 = exact_line_search(A, x_star, x0, d0)
    x1 = x0 + alpha0 * d0

    g1 = A @ (x1 - x_star)
    beta0 = float(g1 @ g1) / float(g0 @ g0)
    d1 = -g1 + beta0 * d0
    alpha1 = exact_line_search(A, x_star, x1, d1)
    x2 = x1 + alpha1 * d1
    return x1, x2, alpha0 * d0, alpha1 * d1


def steepest_descent_history(A: np.ndarray, x_star: np.ndarray, x0: np.ndarray, steps: int) -> list[np.ndarray]:
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(steps):
        g = A @ (x - x_star)
        alpha = float(g @ g) / float(g @ (A @ g))
        x = x - alpha * g
        xs.append(x.copy())
    return xs


def gap_grid(A: np.ndarray, x_star: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    xx1, xx2 = np.meshgrid(x1, x2)
    X = np.stack([xx1 - x_star[0], xx2 - x_star[1]], axis=-1)
    return 0.5 * np.einsum("...i,ij,...j->...", X, A, X)


def square_limits(points: np.ndarray, pad: float) -> tuple[tuple[float, float], tuple[float, float], float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_span = 0.5 * max(maxs - mins) + pad
    return (center[0] - half_span, center[0] + half_span), (center[1] - half_span, center[1] + half_span), half_span


def add_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    lw: float,
    alpha: float = 1.0,
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            posA=tuple(start),
            posB=tuple(end),
            arrowstyle="-|>",
            mutation_scale=11.0,
            linewidth=lw,
            color=color,
            alpha=alpha,
            shrinkA=1.5,
            shrinkB=2.5,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )


def label_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, size: float = 8.5) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        color=color,
        fontsize=size,
        bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.97},
        zorder=7,
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "coordinate-vs-conjugate.pdf"
    out_pgf = out_dir / "coordinate-vs-conjugate.pgf"
    out_svg = out_dir / "coordinate-vs-conjugate.svg"
    out_png = out_dir / "coordinate-vs-conjugate.png"

    # Real quadratic example used for the figure.
    A = rotation(35.0) @ np.diag([0.3, 10.0]) @ rotation(35.0).T
    x_star = np.array([2.0, 1.4], dtype=float)
    x0 = np.array([0.8, 0.3], dtype=float)

    # Real two-step CG path for a 2D SPD quadratic.
    x1, x2, step0, step1 = linear_cg_two_steps(A, x_star, x0)

    # We keep the transformed coordinates tied to the actual conjugate basis,
    # but rescale the basis vectors diagonally to make the diagonalized picture
    # readable on the page while preserving the coordinate-descent structure.
    C = np.column_stack([0.6 * step0, 1.2 * step1])
    D = C.T @ A @ C
    xhat0 = np.linalg.solve(C, x0)
    xhat1 = np.linalg.solve(C, x1)
    xhat2 = np.linalg.solve(C, x2)
    xhat_star = np.linalg.solve(C, x_star)

    # Real steepest-descent history on the same quadratic for the red cue.
    gd_hist = steepest_descent_history(A, x_star, x0, steps=6)

    gap0 = quadratic_gap(A, x_star, x0)
    levels = gap0 * np.array([0.10, 0.24, 0.50, 0.90])

    xhat_all = np.vstack([xhat0, xhat1, xhat2, xhat_star])
    xhat_xlim, xhat_ylim, xhat_half = square_limits(xhat_all, pad=0.35)
    xhat_grid_1 = np.linspace(*xhat_xlim, 420)
    xhat_grid_2 = np.linspace(*xhat_ylim, 380)
    gap_hat = gap_grid(D, xhat_star, xhat_grid_1, xhat_grid_2)

    x_all = np.vstack(gd_hist + [x1, x2, x_star])
    x_xlim, x_ylim, x_half = square_limits(x_all, pad=0.34)
    x_grid_1 = np.linspace(*x_xlim, 420)
    x_grid_2 = np.linspace(*x_ylim, 380)
    gap_x = gap_grid(A, x_star, x_grid_1, x_grid_2)

    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.95, wspace=0.18)

    for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Left panel: exact same iterates in the transformed coordinate system.
    ax0.contour(
        xhat_grid_1,
        xhat_grid_2,
        gap_hat,
        levels=levels,
        colors=[SLATE],
        linewidths=0.9,
        alpha=0.74,
    )
    ax0.set_xlim(*xhat_xlim)
    ax0.set_ylim(*xhat_ylim)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_box_aspect(1.0)
    ax0.set_xlabel(r"$\hat x_1$")
    ax0.set_ylabel(r"$\hat x_2$")
    ax0.scatter([xhat0[0], xhat1[0], xhat2[0]], [xhat0[1], xhat1[1], xhat2[1]], s=14, color=INDIGO, zorder=6)
    add_arrow(ax0, xhat0, xhat1, color=TEAL, lw=1.9, alpha=0.78)
    add_arrow(ax0, xhat1, xhat2, color=TEAL, lw=1.9, alpha=0.78)
    ax0.plot([xhat0[0], xhat1[0]], [xhat0[1], xhat1[1]], ls=(0, (3, 3)), color=SLATE, lw=0.75, alpha=0.30, zorder=1)
    ax0.plot([xhat1[0], xhat1[0]], [xhat1[1], xhat2[1]], ls=(0, (3, 3)), color=SLATE, lw=0.75, alpha=0.30, zorder=1)
    label_box(ax0, (xhat_xlim[0] + 0.72 * (2 * xhat_half), xhat_ylim[0] + 0.88 * (2 * xhat_half)), r"$\hat f(\hat x)$", SLATE, 8.8)
    ax0.annotate(
        "",
        xy=(xhat_xlim[0] + 0.64 * (2 * xhat_half), xhat_ylim[0] + 0.80 * (2 * xhat_half)),
        xytext=(xhat_xlim[0] + 0.57 * (2 * xhat_half), xhat_ylim[0] + 0.73 * (2 * xhat_half)),
        arrowprops={"arrowstyle": "->", "color": SLATE, "lw": 0.8},
    )
    label_box(ax0, (xhat0[0] - 0.10, xhat0[1] - 0.18), r"$\hat x^0$", INDIGO, 8.3)
    label_box(ax0, (xhat1[0] - 0.08, xhat1[1] - 0.18), r"$\hat x^1$", INDIGO, 8.3)
    label_box(ax0, (xhat2[0] - 0.08, xhat2[1] + 0.10), r"$\hat x^2=\hat x^\star$", INDIGO, 8.1)

    # Right panel: mapped-back CG and actual steepest descent on the same f.
    ax1.contour(
        x_grid_1,
        x_grid_2,
        gap_x,
        levels=levels,
        colors=[SLATE],
        linewidths=0.9,
        alpha=0.74,
    )
    ax1.set_xlim(*x_xlim)
    ax1.set_ylim(*x_ylim)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_box_aspect(1.0)
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")

    ax1.scatter([x0[0], x1[0], x2[0]], [x0[1], x1[1], x2[1]], s=14, color=INDIGO, zorder=6)
    add_arrow(ax1, x0, x1, color=TEAL, lw=2.0, alpha=0.78)
    add_arrow(ax1, x1, x2, color=TEAL, lw=2.0, alpha=0.78)

    for start, end in zip(gd_hist[:-1], gd_hist[1:]):
        add_arrow(ax1, start, end, color=CRIMSON, lw=1.15, alpha=0.42, zorder=4)

    label_box(ax1, (x_xlim[0] + 0.14 * (2 * x_half), x_ylim[0] + 0.88 * (2 * x_half)), r"$\mathrm{CG}$", SLATE, 8.8)
    ax1.annotate(
        "",
        xy=(0.52 * x1 + 0.48 * x2),
        xytext=(x_xlim[0] + 0.23 * (2 * x_half), x_ylim[0] + 0.70 * (2 * x_half)),
        arrowprops={"arrowstyle": "->", "color": SLATE, "lw": 0.8},
    )
    label_box(ax1, (x_xlim[0] + 0.73 * (2 * x_half), x_ylim[0] + 0.15 * (2 * x_half)), r"$\mathrm{GD}$", CRIMSON, 8.8)
    ax1.annotate(
        "",
        xy=gd_hist[3] + np.array([0.00, 0.03]),
        xytext=(x_xlim[0] + 0.64 * (2 * x_half), x_ylim[0] + 0.24 * (2 * x_half)),
        arrowprops={"arrowstyle": "->", "color": CRIMSON, "lw": 0.8},
    )
    label_box(ax1, (x_xlim[0] + 0.86 * (2 * x_half), x_ylim[0] + 0.84 * (2 * x_half)), r"$f(x)$", SLATE, 8.8)
    ax1.annotate(
        "",
        xy=(x_xlim[0] + 0.75 * (2 * x_half), x_ylim[0] + 0.80 * (2 * x_half)),
        xytext=(x_xlim[0] + 0.82 * (2 * x_half), x_ylim[0] + 0.82 * (2 * x_half)),
        arrowprops={"arrowstyle": "->", "color": SLATE, "lw": 0.8},
    )
    label_box(ax1, (x0[0] - 0.02, x0[1] - 0.16), r"$x^0$", INDIGO, 8.3)
    label_box(ax1, (x1[0] - 0.23, x1[1] - 0.02), r"$x^1$", INDIGO, 8.3)
    label_box(ax1, (x2[0] + 0.05, x2[1] + 0.04), r"$x^2$", INDIGO, 8.3)
    label_box(ax1, (x_star[0] + 0.35, x_star[1] + 0.35), r"$x^\star$", SLATE, 8.5)
    ax1.annotate(
        "",
        xy=x_star + np.array([0.03, 0.03]),
        xytext=x_star + np.array([0.30, 0.28]),
        arrowprops={"arrowstyle": "->", "color": SLATE, "lw": 0.8},
    )

    fig.savefig(out_pdf)
    fig.savefig(out_pgf)
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=220)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_pgf}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
