#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path

XDG_CACHE_DIR = Path("/tmp/xdg-cache")
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

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
        "figure.figsize": (5.2, 4.8),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SLATE,
        "axes.labelcolor": "black",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
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


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    wrapper_stem = "kkt-disk-example-embed"
    out_pdf = out_dir / "kkt-disk-example-embed.pdf"
    out_svg = out_dir / "kkt-disk-example.svg"
    out_png = out_dir / "kkt-disk-example.png"

    x = np.linspace(-1.15, 1.15, 501)
    y = np.linspace(-1.15, 1.15, 501)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2
    mask = X**2 + Y**2 > 1.0
    Z = np.ma.masked_where(mask, Z)

    fig, ax = plt.subplots(constrained_layout=True)

    levels = np.linspace(-1.0, 1.0, 13)
    contour_fill = ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap="RdBu_r",
        alpha=0.78,
        antialiased=True,
    )
    contour_lines = ax.contour(
        X,
        Y,
        Z,
        levels=np.linspace(-1.0, 1.0, 9),
        colors=SLATE,
        linewidths=0.75,
        alpha=0.55,
    )
    ax.clabel(contour_lines, fmt="%0.1f", inline=True, fontsize=7, colors=SLATE)

    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.fill(np.cos(theta), np.sin(theta), color=TEAL, alpha=0.07, zorder=0)
    ax.plot(np.cos(theta), np.sin(theta), color=SLATE, lw=1.2, zorder=3)

    ax.axhline(0.0, color=SLATE, lw=0.7, alpha=0.45, zorder=1)
    ax.axvline(0.0, color=SLATE, lw=0.7, alpha=0.45, zorder=1)

    ax.scatter([0.0], [0.0], color=CRIMSON, s=28, zorder=5)
    ax.scatter([0.0, 0.0], [1.0, -1.0], color=DEEPBLUE, s=140, marker="*", zorder=6)

    ax.annotate(
        r"$(0,0)$",
        xy=(0.0, 0.0),
        xytext=(0.12, 0.1),
        textcoords="data",
        fontsize=9,
        color=CRIMSON,
    )
    ax.annotate(
        r"$(0,1)$",
        xy=(0.0, 1.0),
        xytext=(0.11, 0.92),
        textcoords="data",
        fontsize=9,
        color=DEEPBLUE,
    )
    ax.annotate(
        r"$(0,-1)$",
        xy=(0.0, -1.0),
        xytext=(0.11, -1.08),
        textcoords="data",
        fontsize=9,
        color=DEEPBLUE,
    )
    ax.text(-0.92, 1.03, r"$x^2+y^2=1$", color=SLATE, fontsize=9)
    ax.text(-1.03, -1.08, r"$f(x,y)=x^2-y^2$", color=SLATE, fontsize=9)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=220)
    write_incfig_wrapper(out_dir, wrapper_stem, out_pdf.name)

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
