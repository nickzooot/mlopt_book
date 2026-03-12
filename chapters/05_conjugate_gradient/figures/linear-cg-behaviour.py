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
        "figure.figsize": (10.0, 4.2),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.labelcolor": "black",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.size": 10,
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    }
)


def write_incfig_wrapper(out_dir: Path, stem: str, pdf_name: str) -> None:
    wrapper = (
        f"%% Minimal wrapper for \\\\incfig{{{stem}}}\n"
        "\\begingroup%\n"
        "  \\makeatletter%\n"
        "  \\ifx\\svgwidth\\undefined%\n"
        "    \\def\\svgwidth{\\columnwidth}%\n"
        "  \\fi%\n"
        "  \\makeatother%\n"
        f"  \\includegraphics[width=\\svgwidth]{{{pdf_name}}}%\n"
        "\\endgroup%\n"
    )
    (out_dir / f"{stem}.pdf_tex").write_text(wrapper, encoding="utf-8")


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


def fit_scale(basis: np.ndarray, values: np.ndarray) -> float:
    return float(basis @ values) / float(basis @ basis)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    wrapper_stem = "linear-cg-behaviour-embed"
    out_pdf = out_dir / "linear-cg-behaviour-embed.pdf"
    out_svg = out_dir / "linear-cg-behaviour.svg"
    out_png = out_dir / "linear-cg-behaviour.png"

    kappas = np.arange(50.0, 3001.0, 150.0)
    dims = [10, 100, 1000]
    colors = {10: CRIMSON, 100: TEAL, 1000: DEEPBLUE}

    gd_curves = {n: averaged_counts("gd", n, kappas, samples=10) for n in dims}
    cg_curves = {n: averaged_counts("cg", n, kappas, samples=10) for n in dims}
    fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True)

    for ax in (ax0, ax1):
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(SLATE)
            ax.spines[side].set_linewidth(0.8)
        ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.6)
        ax.set_xlim(0.0, 3000.0)
        ax.set_xticks([500, 1000, 1500, 2000, 2500, 3000])
        ax.set_xlabel(r"condition number $\kappa$")
        ax.set_ylabel("iterations")

    for n in dims:
        ax0.plot(kappas, gd_curves[n], color=colors[n], lw=1.8, label=fr"$n={n}$")
        ax1.plot(kappas, cg_curves[n], color=colors[n], lw=1.8, label=fr"$n={n}$")
    for n in dims:
        gd_reference = fit_scale(kappas, gd_curves[n]) * kappas
        cg_reference = fit_scale(np.sqrt(kappas), cg_curves[n]) * np.sqrt(kappas)
        ax0.plot(kappas, gd_reference, color=colors[n], lw=1.1, ls="--", alpha=0.65, zorder=1)
        ax1.plot(kappas, cg_reference, color=colors[n], lw=1.1, ls="--", alpha=0.65, zorder=1)

    ax0.set_title("Gradient descent")
    ax1.set_title("Conjugate gradient")
    ax0.set_ylim(0.0, 5600.0)
    ax0.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
    ax1.set_ylim(0.0, 105.0)
    ax1.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax0.legend(loc="upper left", fontsize=8, handlelength=2.4)
    ax1.legend(loc="upper left", fontsize=8, handlelength=2.4)

    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=220)
    write_incfig_wrapper(out_dir, wrapper_stem, out_pdf.name)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
