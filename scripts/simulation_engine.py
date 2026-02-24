"""Core kinematic simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, atan2, acos
from typing import Iterable, Iterator, Sequence

import numpy as np
import sympy as sp
from tqdm import tqdm

from .control import TrajectoryPlanner
from .theoretical_calculations import (
    compute_forward_kinematics,
    compute_jacobian,
    compute_inverse_differential,
    l0,
    l1,
    l2,
    l3,
    q1,
    q2,
    q3,
)


@dataclass
class SimulationConfig:
    T: float = 3.0
    dt: float = 1 / 30.0
    link_lengths: Sequence[float] = (1, 1, 1, 1)
    start: Sequence[float] = (-1, +1, 2.5)
    goal: Sequence[float] = (+1, -1, 2.5)
    elbows: tuple[int, int] = (0, 0)
    show_progress: bool = True


@dataclass
class TimeStepResult:
    t: float
    tau: float
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    qdot: np.ndarray


class RobotSystem:
    """Holds symbolic and numeric representations of the robot."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.link_lengths = np.asarray(config.link_lengths, dtype=float)
        self.A = np.asarray(config.start, dtype=float)
        self.B = np.asarray(config.goal, dtype=float)
        self.elbows = config.elbows
        self._build_symbolics()

    def _build_symbolics(self):
        self.A0_g, self.A1_0, self.A2_1, self.Ae_2, self.Ae_g, self.A1_g, self.A2_g = (
            compute_forward_kinematics()
        )
        J_L, J_A, J = compute_jacobian(
            self.A0_g, self.A1_0, self.A2_1, self.Ae_2, self.Ae_g
        )
        det_JL_factored, adj_JL = compute_inverse_differential(J_L)
        self.inv_JL = (adj_JL / det_JL_factored).subs(self._link_subs)
        self.inv_JL_func = sp.lambdify((q1, q2, q3), self.inv_JL, "numpy")

        matrices = [
            "A0_g",
            "A1_0",
            "A2_1",
            "Ae_2",
            "Ae_g",
            "A1_g",
            "A2_g",
        ]
        for name in matrices:
            setattr(self, name, getattr(self, name).subs(self._link_subs))

        self.transforms = [self.A0_g, self.A1_g, self.A2_g, self.Ae_g]
        self.offsets = [
            sp.Matrix([0, 0, 0]),
            sp.Matrix([0, 0, self.link_lengths[1]]),
            sp.Matrix([0, 0, 0]),
            sp.Matrix([0, 0, 0]),
        ]

    @property
    def _link_subs(self):
        return {
            l0: self.link_lengths[0],
            l1: self.link_lengths[1],
            l2: self.link_lengths[2],
            l3: self.link_lengths[3],
        }

    def inverse_kinematics(self, position: np.ndarray) -> tuple[float, float, float]:
        px, py, pz = position
        l0_val, l1_val, l2_val, l3_val = self.link_lengths

        r = sqrt(px**2 + (pz - l0_val) ** 2)
        phi = atan2(-px, pz - l0_val)
        q1_sol = phi + ((-1) ** self.elbows[0]) * acos(l1_val / r)

        R = px * np.cos(q1_sol) + (pz - l0_val) * np.sin(q1_sol)
        c3 = (R**2 + py**2 - l2_val**2 - l3_val**2) / (2 * l2_val * l3_val)
        s3 = ((-1) ** self.elbows[1]) * sqrt(max(0.0, 1 - c3**2))
        q3_sol = atan2(s3, c3)
        q2_sol = atan2(py, R) - atan2(l3_val * s3, l2_val + l3_val * c3)
        return q1_sol, q2_sol, q3_sol

    def joint_positions(self, q_vals: Sequence[float]) -> np.ndarray:
        angle_subs = {q1: q_vals[0], q2: q_vals[1], q3: q_vals[2]}
        joint_positions = [[0, 0, 0]]
        for T, offset in zip(self.transforms, self.offsets):
            T_num = sp.Matrix(T.subs(angle_subs))
            p_h = T_num * sp.Matrix.vstack(offset, sp.Matrix([1]))
            joint_positions.append(np.array(p_h[:3], dtype=float).flatten())
        return np.vstack(joint_positions)


class KinematicSimulator:
    def __init__(
        self,
        robot: RobotSystem,
        planner: TrajectoryPlanner,
        config: SimulationConfig,
    ):
        self.robot = robot
        self.planner = planner
        self.config = config

    def simulate(self, show_progress: bool | None = None) -> list[TimeStepResult]:
        show_progress = self.config.show_progress if show_progress is None else show_progress
        times = np.arange(0, self.config.T + self.config.dt, self.config.dt)
        iterator: Iterable[float]
        if show_progress:
            iterator = tqdm(times, desc="Simulating", unit="step")
        else:
            iterator = times

        results = []
        for t in iterator:
            result = self._step(t)
            results.append(result)
        return results

    def stream(self, loop: bool = True) -> Iterator[TimeStepResult]:
        while True:
            for t in np.arange(0, self.config.T + self.config.dt, self.config.dt):
                yield self._step(t)
            if not loop:
                break

    def _step(self, t: float) -> TimeStepResult:
        tau, pos, vel = self.planner.evaluate(t)
        q_vals = np.asarray(self.robot.inverse_kinematics(pos), dtype=float)
        inv_JL_val = np.array(self.robot.inv_JL_func(*q_vals), dtype=float)
        qdot_vals = inv_JL_val @ vel
        return TimeStepResult(
            t=float(t),
            tau=float(tau),
            p=pos,
            v=vel,
            q=q_vals,
            qdot=qdot_vals,
        )
