"""Visualization helpers for robot kinematics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection

from .simulation_engine import RobotSystem, SimulationConfig, TimeStepResult
from .theoretical_calculations import q1, q2, q3


def plot_workspace(robot: RobotSystem, config: SimulationConfig, save_path: str | None = None):
    px_expr, py_expr, pz_expr = robot.Ae_g[0, 3], robot.Ae_g[1, 3], robot.Ae_g[2, 3]
    fk_num = sp.lambdify((q1, q2, q3), (px_expr, py_expr, pz_expr), "numpy")

    q1_vals = np.linspace(-np.pi, np.pi, 50)
    q2_vals = np.linspace(-np.pi, np.pi, 25)
    q3_vals = np.linspace(-np.pi, np.pi, 25)

    Q1, Q2, Q3 = np.meshgrid(q1_vals, q2_vals, q3_vals, indexing="ij")
    X, Y, Z = fk_num(Q1, Q2, Q3)

    angle_subs_zero = {q1: 0, q2: 0, q3: 0}
    joint_positions = [[0, 0, 0]]
    for T, offset in zip(robot.transforms, robot.offsets):
        T_num = sp.Matrix(T.subs(angle_subs_zero))
        p_h = T_num * sp.Matrix.vstack(offset, sp.Matrix([1]))
        joint_positions.append(np.array(p_h[:3], dtype=float).flatten())
    joint_positions = np.vstack(joint_positions)

    fig, (ax_xz, ax_xy) = plt.subplots(1, 2, figsize=(10, 5))

    ax_xz.scatter(np.ravel(X), np.ravel(Z), s=1, alpha=0.3)
    ax_xz.plot(joint_positions[:, 0], joint_positions[:, 2], "r-o", label="q=0")
    ax_xz.plot([config.start[0], config.goal[0]], [config.start[2], config.goal[2]], "g-o", label="A-B")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title("Workspace projection (x-z)")
    ax_xz.axis("equal")

    ax_xy.scatter(np.ravel(X), np.ravel(Y), s=1, alpha=0.3)
    ax_xy.plot(joint_positions[:, 0], joint_positions[:, 1], "r-o", label="q=0")
    ax_xy.plot([config.start[0], config.goal[0]], [config.start[1], config.goal[1]], "g-o", label="A-B")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("Workspace projection (x-y)")
    ax_xy.axis("equal")

    ax_xz.legend()
    ax_xy.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_simulation_results(results: Sequence[TimeStepResult], save_path: str | None = None):
    ts = np.array([r.t for r in results])
    p = np.array([r.p for r in results])
    v = np.array([r.v for r in results])
    q = np.array([r.q for r in results])
    qdot = np.array([r.qdot for r in results])

    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex="col")

    labels = [r"$p_x$", r"$p_y$", r"$p_z$"]
    for j in range(3):
        axes[0, j].plot(ts, p[:, j])
        axes[0, j].set_ylabel(labels[j])
        axes[0, j].set_title(f"{labels[j]} vs $t$")
        axes[0, j].grid(True)

    labels_v = [r"$v_x$", r"$v_y$", r"$v_z$"]
    for j in range(3):
        axes[1, j].plot(ts, v[:, j])
        axes[1, j].set_ylabel(labels_v[j])
        axes[1, j].set_title(f"{labels_v[j]} vs $t$")
        axes[1, j].grid(True)

    labels_q = [r"$q_1$", r"$q_2$", r"$q_3$"]
    tick_vals = np.pi * np.array([-1, -0.5, 0, 0.5, 1])
    tick_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    for j in range(3):
        axes[2, j].plot(ts, q[:, j])
        axes[2, j].set_ylabel(labels_q[j])
        axes[2, j].set_yticks(tick_vals)
        axes[2, j].set_yticklabels(tick_labels)
        axes[2, j].set_title(f"{labels_q[j]} vs $t$")
        axes[2, j].grid(True)

    labels_qd = [r"$\dot{q}_1$", r"$\dot{q}_2$", r"$\dot{q}_3$"]
    for j in range(3):
        axes[3, j].plot(ts, qdot[:, j])
        axes[3, j].set_ylabel(labels_qd[j])
        axes[3, j].set_title(f"{labels_qd[j]} vs $t$")
        axes[3, j].set_xlabel(r"$t$")
        axes[3, j].grid(True)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def create_motion_gif(
    results: Sequence[TimeStepResult],
    robot: RobotSystem,
    config: SimulationConfig,
    save_path: str | None = None,
    sample_path: str | None = None,
):
    joint_trajs = [robot.joint_positions(r.q) for r in results]
    all_points = np.vstack(joint_trajs)
    xs, ys, zs = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min())
    mid = np.array([xs.mean(), ys.mean(), zs.mean()])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot([], [], [], "bo-", lw=2)

    _set_axes(ax, mid, max_range)
    _add_targets(ax, config.start, config.goal)

    def update(frame_idx):
        pts = joint_trajs[frame_idx]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        return line,

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(joint_trajs),
        interval=1000 * config.dt,
        blit=False,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer="pillow", fps=int(1.0 / config.dt))

    _plot_samples(joint_trajs, config, mid, max_range, sample_path)
    plt.show()


def _plot_samples(joint_trajs, config, mid, max_range, sample_path):
    sample_taus = [0.0, 0.3, 0.5, 0.8]
    n_frames = len(joint_trajs)
    sample_indices = [int(tau * (n_frames - 1)) for tau in sample_taus]
    colors = ["r", "g", "b", "m"]

    fig_samples = plt.figure(figsize=(6, 6))
    ax_samples = fig_samples.add_subplot(111, projection="3d")

    for tau_val, idx, color in zip(sample_taus, sample_indices, colors):
        pts = joint_trajs[idx]
        ax_samples.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker="o", color=color, label=f"$\\tau = {tau_val:.1f}$")

    _set_axes(ax_samples, mid, max_range)
    _add_targets(ax_samples, config.start, config.goal)
    ax_samples.set_title("Robot configurations at $\\tau=0,0.3,0.5,0.8$")
    ax_samples.legend()

    if sample_path:
        Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
        fig_samples.savefig(sample_path, bbox_inches="tight")


def _set_axes(ax, mid, max_range):
    half = max_range / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Robot movement in 3D")


def _add_targets(ax, start, goal):
    ax.scatter(start[0], start[1], start[2], c="r", marker="o", label="A")
    ax.scatter(goal[0], goal[1], goal[2], c="g", marker="o", label="B")
    ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], "k--", label="A-B")
