#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.linalg as la


DEEPBLUE = "#2c3e7c"
CRIMSON = "#c23b22"
SLATE = "#47515c"
GREEN = "#2a9d8f"
PURPLE = "#7B2CBF"

LINE_ALPHA = 0.85


def time_call(fn, *, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        best = min(best, perf_counter() - t0)
    return best


def repeats_for_n(n: int, *, base: int = 3) -> int:
    # Large n are expensive: avoid multiple repeats there.
    if n <= 2000:
        return base
    if n <= 5000:
        return 2
    return 1


def time_factorization(factory, fn, *, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        A = factory()
        t0 = perf_counter()
        fn(A)
        best = min(best, perf_counter() - t0)
    return best


def make_spd_kernel_matrix(n: int, *, ell: float = 25.0, jitter: float = 1e-3) -> np.ndarray:
    # Dense SPD Toeplitz matrix: A_ij = exp(-|i-j|/ell) + jitter * 1{i=j}.
    # This is a valid covariance kernel on a 1D grid; jitter improves conditioning.
    c = np.exp(-np.arange(n, dtype=float) / float(ell))
    A = np.empty((n, n), dtype=float, order="F")
    for j in range(n):
        A[j:, j] = c[: n - j]
        if j:
            A[:j, j] = c[1 : j + 1][::-1]
    A.flat[:: n + 1] += float(jitter)
    return A


def main() -> None:
    # Avoid matplotlib cache warnings on machines where $HOME is read-only.
    cache_dir = Path("/tmp/matplotlib-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "factorization-cost-vs-n.pdf"
    out_pdf_tex = out_dir / "factorization-cost-vs-n.pdf_tex"
    out_svg = out_dir / "factorization-cost-vs-n.svg"

    # Benchmark: factorization runtimes vs dimension.
    n_values = (
        list(range(5, 101, 5))
        + list(range(150, 501, 50))
        + list(range(600, 2001, 200))
        + [2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    )
    repeats_base = 3

    # Warm-up (loads LAPACK/BLAS, builds caches).
    A0 = make_spd_kernel_matrix(200)
    la.cho_factor(A0, lower=True, check_finite=False, overwrite_a=True)
    la.ldl(A0, lower=True, hermitian=True, check_finite=False, overwrite_a=True)
    la.lu_factor(A0, check_finite=False, overwrite_a=True)
    la.qr(A0, mode="economic", check_finite=False, overwrite_a=True)
    la.eigh(A0, check_finite=False, overwrite_a=True)
    del A0

    times_chol: list[float] = []
    times_ldl: list[float] = []
    times_lu: list[float] = []
    times_qr: list[float] = []
    times_eig: list[float] = []

    for n in n_values:
        repeats = repeats_for_n(n, base=repeats_base)
        t_chol = time_factorization(
            lambda: make_spd_kernel_matrix(n),
            lambda A: la.cho_factor(A, lower=True, check_finite=False, overwrite_a=True),
            repeats=repeats,
        )
        t_ldl = time_factorization(
            lambda: make_spd_kernel_matrix(n),
            lambda A: la.ldl(A, lower=True, hermitian=True, check_finite=False, overwrite_a=True),
            repeats=repeats,
        )
        t_lu = time_factorization(
            lambda: make_spd_kernel_matrix(n),
            lambda A: la.lu_factor(A, check_finite=False, overwrite_a=True),
            repeats=repeats,
        )
        t_qr = time_factorization(
            lambda: make_spd_kernel_matrix(n),
            lambda A: la.qr(A, mode="economic", check_finite=False, overwrite_a=True),
            repeats=repeats,
        )
        t_eig = time_factorization(
            lambda: make_spd_kernel_matrix(n),
            lambda A: la.eigh(A, check_finite=False, overwrite_a=True),
            repeats=repeats,
        )

        times_chol.append(t_chol)
        times_ldl.append(t_ldl)
        times_lu.append(t_lu)
        times_qr.append(t_qr)
        times_eig.append(t_eig)

        print(
            f"n={n:5d}: chol={t_chol:.4f}s  ldl={t_ldl:.4f}s  lu={t_lu:.4f}s  qr={t_qr:.4f}s  eig={t_eig:.4f}s"
        )

    n_arr = np.array(n_values, dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    ax.plot(n_arr, times_chol, lw=2.0, color=DEEPBLUE, alpha=LINE_ALPHA, label="Cholesky")
    ax.plot(n_arr, times_ldl, lw=2.0, color=PURPLE, alpha=LINE_ALPHA, label="LDL")
    ax.plot(n_arr, times_lu, lw=2.0, color=GREEN, alpha=LINE_ALPHA, label="LU")
    ax.plot(n_arr, times_qr, lw=2.0, color=CRIMSON, alpha=LINE_ALPHA, label="QR")
    ax.plot(n_arr, times_eig, lw=2.0, color=SLATE, alpha=LINE_ALPHA, label="Eig (symmetric)")

    ax.set_xlabel(r"matrix dimension $n$")
    ax.set_ylabel("time (seconds)")
    ax.set_title(f"Dense factorization runtime vs dimension (best of up to {repeats_base} runs)")
    ax.grid(True, which="both", ls=":", lw=0.7, alpha=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_yscale("log")

    fig.savefig(out_pdf)
    fig.savefig(out_svg)

    out_pdf_tex.write_text(
        "%% Minimal wrapper for \\\\incfig{factorization-cost-vs-n}\n"
        "\\\\begingroup%\n"
        "  \\\\makeatletter%\n"
        "  \\\\ifx\\\\svgwidth\\\\undefined%\n"
        "    \\\\def\\\\svgwidth{\\\\columnwidth}%\n"
        "  \\\\fi%\n"
        "  \\\\makeatother%\n"
        "  \\\\includegraphics[width=\\\\svgwidth]{factorization-cost-vs-n.pdf}%\n"
        "\\\\endgroup%\n"
    )

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_svg}")
    print(f"Wrote {out_pdf_tex}")


if __name__ == "__main__":
    main()
