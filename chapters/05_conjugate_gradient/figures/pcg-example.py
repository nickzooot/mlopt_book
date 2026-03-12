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
SLATE = "#47515c"


plt.rcParams.update(
    {
        "figure.figsize": (5.4, 3.4),
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
        "font.family": "sans-serif",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
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


def build_note_matrix(n: int = 500) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    idx = np.arange(1, n + 1, dtype=float)
    A[np.arange(n), np.arange(n)] = 1.0 + idx**1.2
    for offset in (1, 100):
        i = np.arange(n - offset)
        A[i, i + offset] = 1.0
        A[i + offset, i] = 1.0
    return A


def cg_history(A: np.ndarray, b: np.ndarray, max_iters: int = 170) -> np.ndarray:
    x = np.zeros_like(b)
    g = A @ x - b
    d = -g.copy()
    hist = [np.linalg.norm(g)]
    for _ in range(max_iters):
        Ad = A @ d
        alpha = float(g @ g) / float(d @ Ad)
        x = x + alpha * d
        g_next = g + alpha * Ad
        hist.append(np.linalg.norm(g_next))
        if hist[-1] <= 1e-14:
            break
        beta = float(g_next @ g_next) / float(g @ g)
        d = -g_next + beta * d
        g = g_next
    return np.asarray(hist)


def pcg_history(A: np.ndarray, b: np.ndarray, M_inv: np.ndarray, max_iters: int = 170) -> np.ndarray:
    x = np.zeros_like(b)
    g = A @ x - b
    z = M_inv * g
    d = -z.copy()
    hist = [np.linalg.norm(g)]
    for _ in range(max_iters):
        Ad = A @ d
        gz = float(g @ z)
        alpha = gz / float(d @ Ad)
        x = x + alpha * d
        g_next = g + alpha * Ad
        hist.append(np.linalg.norm(g_next))
        if hist[-1] <= 1e-14:
            break
        z_next = M_inv * g_next
        beta = float(g_next @ z_next) / gz
        d = -z_next + beta * d
        g = g_next
        z = z_next
    return np.asarray(hist)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    wrapper_stem = "pcg-example-embed"
    out_pdf = out_dir / "pcg-example-embed.pdf"
    out_svg = out_dir / "pcg-example.svg"
    out_png = out_dir / "pcg-example.png"

    A = build_note_matrix()
    b = np.ones(A.shape[0], dtype=float)
    M_inv = 1.0 / np.diag(A)

    cg = cg_history(A, b)
    pcg = pcg_history(A, b, M_inv)

    cg /= cg[0]
    pcg /= pcg[0]

    fig, ax = plt.subplots(constrained_layout=True)

    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.6)

    ax.semilogy(np.arange(len(cg)), cg, color=CRIMSON, lw=1.8, label="CG")
    ax.semilogy(np.arange(len(pcg)), pcg, color=DEEPBLUE, lw=2.0, label="PCG")
    ax.set_xlim(0.0, 170.0)
    ax.set_ylim(5e-7, 2.0)
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.set_title("CG vs preconditioned CG")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative residual")
    ax.legend(loc="upper right", fontsize=8.0, handlelength=2.4)

    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=220)
    write_incfig_wrapper(out_dir, wrapper_stem, out_pdf.name)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
