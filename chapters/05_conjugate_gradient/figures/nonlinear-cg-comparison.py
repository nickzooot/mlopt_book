#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import warnings

MPL_CACHE_DIR = Path("/tmp/matplotlib-cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import line_search, minimize


DEEPBLUE = "#2c3e7c"
CRIMSON = "#c23b22"
TEAL = "#168f89"
SLATE = "#47515c"
FOREST = "#2f6a3d"
SOFTBLUE = "#4f6fa8"


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


def rosenbrock(x: np.ndarray) -> float:
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            2.0 * (x[0] - 1.0) - 400.0 * x[0] * (x[1] - x[0] ** 2),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=float,
    )


def build_logistic_problem(seed: int = 7) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    m, n = 700, 40
    Z = rng.normal(scale=0.55, size=(m, n))
    scales = np.geomspace(1.0, 4.0, n)
    A = Z * scales
    lam = 5.0e-3
    return A, lam


def logistic_fun_grad(A: np.ndarray, lam: float):
    def fun(x: np.ndarray) -> float:
        Ax = A @ x
        return float(np.sum(np.logaddexp(0.0, Ax)) + 0.5 * lam * (x @ x))

    def grad(x: np.ndarray) -> np.ndarray:
        Ax = A @ x
        sigma = 1.0 / (1.0 + np.exp(-Ax))
        return A.T @ sigma + lam * x

    return fun, grad


@dataclass
class CountingObjective:
    fun: callable
    grad: callable
    f_calls: int = 0
    g_calls: int = 0

    def f(self, x: np.ndarray) -> float:
        self.f_calls += 1
        return float(self.fun(x))

    def g(self, x: np.ndarray) -> np.ndarray:
        self.g_calls += 1
        return np.asarray(self.grad(x), dtype=float)


def strong_wolfe_step(
    obj: CountingObjective,
    x: np.ndarray,
    d: np.ndarray,
    g: np.ndarray,
    f0: float,
    *,
    c1: float = 1e-4,
    c2: float = 0.1,
) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha, *_ = line_search(obj.f, obj.g, x, d, gfk=g, old_fval=f0, c1=c1, c2=c2, maxiter=40)
    if alpha is not None and np.isfinite(alpha) and alpha > 0.0:
        return float(alpha)

    alpha = 1.0
    slope0 = float(g @ d)
    while obj.f(x + alpha * d) > f0 + c1 * alpha * slope0:
        alpha *= 0.5
        if alpha < 1e-12:
            break
    return float(alpha)


def run_gd(
    fun: callable,
    grad: callable,
    x0: np.ndarray,
    f_star: float,
    *,
    max_iters: int = 120,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    obj = CountingObjective(fun, grad)
    x = x0.copy()
    f_hist = [obj.f(x) - f_star]
    eval_hist = [obj.f_calls]
    for _ in range(max_iters):
        g = obj.g(x)
        if np.linalg.norm(g) <= tol:
            break
        d = -g
        alpha = strong_wolfe_step(obj, x, d, g, f_hist[-1] + f_star)
        x = x + alpha * d
        f_hist.append(obj.f(x) - f_star)
        eval_hist.append(obj.f_calls)
    return np.asarray(eval_hist), np.asarray(f_hist)


def run_ncg(
    fun: callable,
    grad: callable,
    x0: np.ndarray,
    f_star: float,
    beta_kind: str,
    *,
    restart: bool = False,
    max_iters: int = 120,
    tol: float = 1e-8,
    nu: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    obj = CountingObjective(fun, grad)
    x = x0.copy()
    g = obj.g(x)
    d = -g.copy()
    f_hist = [obj.f(x) - f_star]
    eval_hist = [obj.f_calls]

    for _ in range(max_iters):
        if np.linalg.norm(g) <= tol:
            break
        f0 = f_hist[-1] + f_star
        alpha = strong_wolfe_step(obj, x, d, g, f0)
        x = x + alpha * d
        g_next = obj.g(x)
        f_hist.append(obj.f(x) - f_star)
        eval_hist.append(obj.f_calls)
        if np.linalg.norm(g_next) <= tol:
            break

        y = g_next - g
        if beta_kind == "fr":
            beta = float(g_next @ g_next) / float(g @ g)
        elif beta_kind == "pr":
            beta = float(g_next @ y) / float(g @ g)
        else:
            raise ValueError(beta_kind)

        if restart and abs(float(g_next @ g)) > nu * float(g_next @ g_next):
            beta = 0.0

        d = -g_next + beta * d
        if float(d @ g_next) >= -1e-12:
            d = -g_next
        g = g_next

    return np.asarray(eval_hist), np.asarray(f_hist)


def logistic_reference_value(fun: callable, grad: callable, n: int) -> float:
    res = minimize(fun, np.zeros(n), jac=grad, method="L-BFGS-B", options={"maxiter": 800})
    return float(res.fun)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    wrapper_stem = "nonlinear-cg-comparison-embed"
    out_pdf = out_dir / "nonlinear-cg-comparison-embed.pdf"
    out_svg = out_dir / "nonlinear-cg-comparison.svg"
    out_png = out_dir / "nonlinear-cg-comparison.png"

    x0_ros = np.array([-1.2, 1.0], dtype=float)
    fr_evals, fr_gap = run_ncg(rosenbrock, rosenbrock_grad, x0_ros, 0.0, "fr", restart=False, max_iters=120)
    frr_evals, frr_gap = run_ncg(rosenbrock, rosenbrock_grad, x0_ros, 0.0, "fr", restart=True, max_iters=120)
    pr_evals, pr_gap = run_ncg(rosenbrock, rosenbrock_grad, x0_ros, 0.0, "pr", restart=False, max_iters=120)

    A, lam = build_logistic_problem()
    fun_log, grad_log = logistic_fun_grad(A, lam)
    f_star_log = logistic_reference_value(fun_log, grad_log, A.shape[1])
    x0_log = np.zeros(A.shape[1])
    gd_evals, gd_gap = run_gd(fun_log, grad_log, x0_log, f_star_log, max_iters=35)
    prlog_evals, prlog_gap = run_ncg(fun_log, grad_log, x0_log, f_star_log, "pr", restart=False, max_iters=35)

    fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True)
    gap_floor = 1e-12

    for ax in (ax0, ax1):
        ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.6)
        ax.set_xlabel("function evaluations")
        ax.set_ylabel(r"$f(x_k)-f^\star$")

    ax0.semilogy(fr_evals, np.maximum(fr_gap, gap_floor), color=FOREST, lw=1.8, label="FR")
    ax0.semilogy(frr_evals, np.maximum(frr_gap, gap_floor), color=CRIMSON, lw=2.0, label="FR + restart")
    ax0.semilogy(pr_evals, np.maximum(pr_gap, gap_floor), color=DEEPBLUE, lw=1.8, label="PR")
    ax0.set_xlim(0.0, 320.0)
    ax0.set_ylim(1e-15, 2.0e2)
    ax0.set_xticks([50, 100, 150, 200, 250, 300])
    ax0.set_title("Rosenbrock")
    ax0.legend(loc="lower left", fontsize=8.0, handlelength=2.4)

    ax1.semilogy(gd_evals, np.maximum(gd_gap, gap_floor), color=SOFTBLUE, lw=1.8, label="GD")
    ax1.semilogy(prlog_evals, np.maximum(prlog_gap, gap_floor), color=TEAL, lw=2.0, label="PR")
    ax1.set_xlim(0.0, 390.0)
    ax1.set_ylim(1e-12, 2.0e1)
    ax1.set_xticks([50, 100, 150, 200, 250, 300, 350])
    ax1.set_title(r"logistic + $\ell_2$")
    ax1.legend(loc="upper right", fontsize=8.0, handlelength=2.4)

    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=220)
    write_incfig_wrapper(out_dir, wrapper_stem, out_pdf.name)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
