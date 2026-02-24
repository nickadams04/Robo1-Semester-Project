"""Trajectory planning utilities."""

from dataclasses import dataclass
import numpy as np


@dataclass
class TrajectoryPlanner:
    """Simple quintic (via-point) planner for end-effector motion."""

    T: float
    start: np.ndarray
    goal: np.ndarray

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float).ravel()
        self.goal = np.asarray(self.goal, dtype=float).ravel()

    def _sigma(self, tau: float) -> tuple[float, float]:
        # Quintic polynomial with zero velocity at endpoints
        s = 16 * tau**2 * (tau - 1) ** 2
        s_dot = 32 * tau * (2 * tau**2 - 3 * tau + 1)
        return s, s_dot

    def evaluate(self, t: float) -> tuple[float, np.ndarray, np.ndarray]:
        tau = np.clip(t / self.T, 0.0, 1.0)
        s, s_dot = self._sigma(tau)
        delta = self.goal - self.start
        position = self.start + s * delta
        velocity = (s_dot / self.T) * delta
        return tau, position, velocity
