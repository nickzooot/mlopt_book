#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


DEEPBLUE = "#2c3e7c"
CRIMSON = "#c23b22"
SLATE = "#47515c"


def rosenbrock(z: np.ndarray) -> float:
    x, y = float(z[0]), float(z[1])
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def grad(z: np.ndarray) -> np.ndarray:
    x, y = float(z[0]), float(z[1])
    return np.array([2.0 * (x - 1.0) - 400.0 * x * (y - x**2), 200.0 * (y - x**2)], dtype=float)


def hess(z: np.ndarray) -> np.ndarray:
    x, y = float(z[0]), float(z[1])
    return np.array([[2.0 - 400.0 * y + 1200.0 * x**2, -400.0 * x], [-400.0 * x, 200.0]], dtype=float)


def backtracking(
    z: np.ndarray,
    d: np.ndarray,
    *,
    f0: float | None = None,
    g0: np.ndarray | None = None,
    c1: float = 1e-4,
    beta: float = 0.5,
    alpha0: float = 1.0,
) -> float:
    if f0 is None:
        f0 = rosenbrock(z)
    if g0 is None:
        g0 = grad(z)
    alpha = float(alpha0)
    while rosenbrock(z + alpha * d) > f0 + c1 * alpha * float(g0.dot(d)):
        alpha *= beta
        if alpha < 1e-12:
            break
    return alpha


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "rosenbrock-newton-vs-gd.pdf"

    z0 = np.array([-1.2, 1.0], dtype=float)

    # Gradient descent (fixed small step, to highlight slow zig-zagging).
    z = z0.copy()
    traj_gd = [z.copy()]
    vals_gd = [rosenbrock(z)]
    alpha_gd = 1e-3
    gd_max_iters = 2000
    for _ in range(gd_max_iters):
        z = z - alpha_gd * grad(z)
        traj_gd.append(z.copy())
        vals_gd.append(rosenbrock(z))
    traj_gd = np.array(traj_gd)
    vals_gd = np.array(vals_gd)

    # Damped Newton (Armijo backtracking).
    z = z0.copy()
    traj_nt = [z.copy()]
    vals_nt = [rosenbrock(z)]
    newton_max_iters = 50
    newton_tol = 1e-10
    for _ in range(newton_max_iters):
        g = grad(z)
        H = hess(z)
        d = np.linalg.solve(H, -g)
        a = backtracking(z, d, f0=rosenbrock(z), g0=g, alpha0=1.0)
        z = z + a * d
        traj_nt.append(z.copy())
        vals_nt.append(rosenbrock(z))
        if np.linalg.norm(g) < newton_tol:
            break
    traj_nt = np.array(traj_nt)
    vals_nt = np.array(vals_nt)

    # Grid for contours.
    x = np.linspace(-2.0, 2.0, 400)
    y = np.linspace(-1.0, 3.0, 400)
    X, Y = np.meshgrid(x, y)
    Z = (1.0 - X) ** 2 + 100.0 * (Y - X**2) ** 2
    levels = np.logspace(-1, 3, 12)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    # Contour + trajectories.
    ax0.contour(X, Y, Z, levels=levels, norm=LogNorm(), colors=SLATE, linewidths=0.8, alpha=0.55)
    ax0.plot(
        traj_gd[:, 0],
        traj_gd[:, 1],
        color=DEEPBLUE,
        lw=1.5,
        label=f"GD (step={alpha_gd:g}, {len(traj_gd) - 1} iters)",
    )
    ax0.plot(traj_nt[:, 0], traj_nt[:, 1], color=CRIMSON, lw=2.0, label=f"Newton (backtracking, {len(traj_nt)-1} iters)")
    ax0.scatter([z0[0]], [z0[1]], c="black", s=20, zorder=5)
    ax0.scatter([1.0], [1.0], c="#7B2CBF", s=200, marker="*", zorder=6)
    ax0.set_xlim(-2.0, 2.0)
    ax0.set_ylim(-1.0, 3.0)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_title("Trajectories on Rosenbrock contours")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.legend(frameon=False, fontsize=8, loc="upper right")

    # Convergence curves.
    ax1.semilogy(vals_gd, color=DEEPBLUE, lw=1.5)
    ax1.semilogy(vals_nt, color=CRIMSON, lw=2.0)
    ax1.set_title("Objective value vs iteration")
    ax1.set_xlabel(r"iteration $k$")
    ax1.set_ylabel(r"$f(x_k,y_k)$")
    ax1.grid(True, which="both", ls=":", lw=0.7, alpha=0.6)

    fig.savefig(out_pdf)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
