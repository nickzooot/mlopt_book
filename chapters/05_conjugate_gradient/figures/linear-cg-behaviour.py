#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE_DIR = Path("/tmp/matplotlib-cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np


DEEPBLUE = "#2c3e7c"
CRIMSON = "#c23b22"
TEAL = "#168f89"
SLATE = "#47515c"


plt.rcParams.update(
    {
        "figure.figsize": (6.1, 3.0),
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
        "legend.frameon": False,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    }
)


def make_diagonal_problem(n: int, kappa: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    diag = np.empty(n, dtype=float)
    diag[0] = 1.0
    if n > 2:
        diag[1:-1] = rng.uniform(1.0, kappa, size=n - 2)
    if n > 1:
        diag[-1] = kappa
    b = rng.normal(size=n)
    return diag, b


def gd_iterations(diag: np.ndarray, b: np.ndarray, tol: float = 5e-3, max_iters: int = 6000) -> int:
    x = np.zeros_like(b)
    rhs_norm = np.linalg.norm(b)
    for k in range(max_iters + 1):
        g = diag * x - b
        if np.linalg.norm(g) <= tol * max(1.0, rhs_norm):
            return k
        alpha = float(g @ g) / float(g @ (diag * g))
        x = x - alpha * g
    return max_iters


def cg_iterations(diag: np.ndarray, b: np.ndarray, tol: float = 5e-3, max_iters: int = 2000) -> int:
    x = np.zeros_like(b)
    g = diag * x - b
    d = -g.copy()
    rhs_norm = np.linalg.norm(b)
    for k in range(max_iters + 1):
        if np.linalg.norm(g) <= tol * max(1.0, rhs_norm):
            return k
        Ad = diag * d
        alpha = float(g @ g) / float(d @ Ad)
        x = x + alpha * d
        g_next = g + alpha * Ad
        beta = float(g_next @ g_next) / float(g @ g)
        d = -g_next + beta * d
        g = g_next
    return max_iters


def averaged_counts(method: str, n: int, kappas: np.ndarray, samples: int = 8) -> np.ndarray:
    counts = []
    for kappa in kappas:
        local = []
        for seed in range(samples):
            rng = np.random.default_rng(10_000 * (seed + 1) + 97 * n + int(round(kappa)))
            diag, b = make_diagonal_problem(n, float(kappa), rng)
            if method == "gd":
                local.append(gd_iterations(diag, b))
            elif method == "cg":
                local.append(cg_iterations(diag, b))
            else:
                raise ValueError(method)
        counts.append(float(np.mean(local)))
    return np.asarray(counts)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "linear-cg-behaviour.pdf"
    out_svg = out_dir / "linear-cg-behaviour.svg"
    out_png = out_dir / "linear-cg-behaviour.png"

    kappas = np.arange(50.0, 1251.0, 100.0)
    dims = [10, 100, 1000]
    colors = {10: CRIMSON, 100: TEAL, 1000: DEEPBLUE}

    gd_curves = {n: averaged_counts("gd", n, kappas) for n in dims}
    cg_curves = {n: averaged_counts("cg", n, kappas) for n in dims}

    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.95, wspace=0.17)

    for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", which="major", ls=":", lw=0.55, alpha=0.28)
        ax.set_xlim(0.0, 1250.0)
        ax.set_xticks([200, 600, 1000])
        ax.set_xlabel(r"condition number $\kappa$")
        ax.set_ylabel("iterations")

    for n in dims:
        ax0.plot(kappas, gd_curves[n], color=colors[n], lw=1.9, label=fr"$n={n}$")
        ax1.plot(kappas, cg_curves[n], color=colors[n], lw=1.9, label=fr"$n={n}$")

    ax0.text(0.03, 0.96, "GD", transform=ax0.transAxes, ha="left", va="top", color=SLATE, fontsize=8.5)
    ax1.text(0.03, 0.96, "CG", transform=ax1.transAxes, ha="left", va="top", color=SLATE, fontsize=8.5)
    ax0.set_ylim(0.0, 3200.0)
    ax0.set_yticks([0, 1000, 2000, 3000])
    ax1.set_ylim(0.0, 80.0)
    ax1.set_yticks([0, 20, 40, 60])
    ax1.legend(loc="lower right", fontsize=7.8, handlelength=2.4, borderaxespad=0.3)

    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=220)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
