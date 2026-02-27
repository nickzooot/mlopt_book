#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from time import perf_counter

# ---------------------------------------------------------------------------
# Benchmark hygiene: force single-threaded BLAS/LAPACK.
#
# We want the plot to reflect algorithmic constant factors, not fluctuations
# from BLAS thread scheduling. On macOS we typically link against Accelerate,
# which is controlled by VECLIB_* variables.
# ---------------------------------------------------------------------------
FORCED_NUM_THREADS = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = FORCED_NUM_THREADS
os.environ["VECLIB_MINIMUM_THREADS"] = FORCED_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = FORCED_NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = FORCED_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = FORCED_NUM_THREADS
os.environ["BLIS_NUM_THREADS"] = FORCED_NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = FORCED_NUM_THREADS
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

import numpy as np
import scipy.linalg as la


DEEPBLUE = "#2c3e7c"
CRIMSON = "#c23b22"
SLATE = "#47515c"
GREEN = "#2a9d8f"
PURPLE = "#7B2CBF"

LINE_ALPHA = 0.85


def time_method(fn, A: np.ndarray) -> float:
    # Time only the factorization call itself (matrix generation/copies happen outside).
    gc_was_enabled = gc.isenabled()
    gc.disable()
    t0 = perf_counter()
    try:
        fn(A)
    finally:
        if gc_was_enabled:
            gc.enable()
    return perf_counter() - t0


def make_spd_gram_matrix(
    n: int,
    rng: np.random.Generator,
    *,
    jitter: float,
) -> np.ndarray:
    # Dense SPD Gram matrix:
    #   G = A^T A + jitter * I,  where A_ij ~ N(0, 1).
    A = rng.standard_normal((n, n), dtype=float)
    A = np.asfortranarray(A)

    G = np.empty((n, n), dtype=float, order="F")
    la.blas.dgemm(alpha=1.0, a=A, b=A, trans_a=True, beta=0.0, c=G, overwrite_c=1)
    G.flat[:: n + 1] += float(jitter)
    return G


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    # Avoid matplotlib cache warnings on machines where $HOME is read-only.
    cache_dir = Path("/tmp/matplotlib-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "factorization-cost-vs-n.pdf"
    out_pdf_tex = out_dir / "factorization-cost-vs-n.pdf_tex"
    out_svg = out_dir / "factorization-cost-vs-n.svg"

    trials = int(os.environ.get("FACTORIZATION_TRIALS", "10"))
    seed = int(os.environ.get("FACTORIZATION_SEED", "0"))
    jitter = float(os.environ.get("FACTORIZATION_JITTER", "1e-3"))
    max_n = int(os.environ.get("FACTORIZATION_MAX_N", "10000"))

    print(  # noqa: T201
        "Thread settings (forced for reproducibility): "
        + ", ".join(
            f"{k}={os.environ.get(k)}"
            for k in (
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        )
    )
    print(  # noqa: T201
        f"Benchmark settings: trials={trials}, seed={seed}, jitter={jitter:g}, max_n={max_n}",
    )

    rng = np.random.default_rng(seed)

    # Benchmark: factorization runtimes vs dimension.
    n_values = (
        list(range(5, 101, 5))
        + list(range(150, 501, 50))
        + list(range(600, 2001, 200))
        + [2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    )
    n_values = [n for n in n_values if n <= max_n]

    # Warm-up (loads LAPACK/BLAS, builds caches).
    rng_warm = np.random.default_rng(seed + 1)
    G0 = make_spd_gram_matrix(200, rng_warm, jitter=jitter)
    la.cho_factor(G0.copy(order="F"), lower=True, check_finite=False, overwrite_a=True)
    la.ldl(G0.copy(order="F"), lower=True, hermitian=True, check_finite=False, overwrite_a=True)
    la.lu_factor(G0.copy(order="F"), check_finite=False, overwrite_a=True)
    la.qr(G0.copy(order="F"), mode="economic", check_finite=False, overwrite_a=True)
    la.eigh(G0.copy(order="F"), check_finite=False, overwrite_a=True)
    del G0

    times_chol: list[float] = []
    times_ldl: list[float] = []
    times_lu: list[float] = []
    times_qr: list[float] = []
    times_eig: list[float] = []

    for n in n_values:
        trial_chol: list[float] = []
        trial_ldl: list[float] = []
        trial_lu: list[float] = []
        trial_qr: list[float] = []
        trial_eig: list[float] = []

        for _ in range(trials):
            # Generate one test matrix per trial and run *all* factorizations on
            # copies of the same matrix (so every method sees identical inputs).
            G = make_spd_gram_matrix(n, rng, jitter=jitter)

            A = G.copy(order="F")
            trial_chol.append(
                time_method(lambda X: la.cho_factor(X, lower=True, check_finite=False, overwrite_a=True), A)
            )

            A = G.copy(order="F")
            trial_ldl.append(
                time_method(lambda X: la.ldl(X, lower=True, hermitian=True, check_finite=False, overwrite_a=True), A)
            )

            A = G.copy(order="F")
            trial_lu.append(time_method(lambda X: la.lu_factor(X, check_finite=False, overwrite_a=True), A))

            A = G.copy(order="F")
            trial_qr.append(time_method(lambda X: la.qr(X, mode="economic", check_finite=False, overwrite_a=True), A))

            A = G.copy(order="F")
            trial_eig.append(time_method(lambda X: la.eigh(X, check_finite=False, overwrite_a=True), A))

        t_chol = float(np.mean(trial_chol))
        t_ldl = float(np.mean(trial_ldl))
        t_lu = float(np.mean(trial_lu))
        t_qr = float(np.mean(trial_qr))
        t_eig = float(np.mean(trial_eig))

        times_chol.append(t_chol)
        times_ldl.append(t_ldl)
        times_lu.append(t_lu)
        times_qr.append(t_qr)
        times_eig.append(t_eig)

        print(  # noqa: T201
            f"n={n:5d} (mean of {trials}): chol={t_chol:.4f}s  ldl={t_ldl:.4f}s  lu={t_lu:.4f}s"
            f"  qr={t_qr:.4f}s  eig={t_eig:.4f}s"
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
    ax.set_title(f"Dense factorization runtime vs dimension (mean of {trials} trials)")
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
