"""High-level orchestration helpers for the kinematic simulation."""

from __future__ import annotations

from pathlib import Path

from .control import TrajectoryPlanner
from .simulation_engine import (
    KinematicSimulator,
    RobotSystem,
    SimulationConfig,
    TimeStepResult,
)
from .visualization import create_motion_gif, plot_simulation_results, plot_workspace


class RobotSimulation:
    """Convenience wrapper that bundles robot model, planner, and simulator."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.robot = RobotSystem(self.config)
        self.planner = TrajectoryPlanner(self.config.T, self.config.start, self.config.goal)
        self.engine = KinematicSimulator(self.robot, self.planner, self.config)
        self.results: list[TimeStepResult] | None = None

    def run(self, show_progress: bool | None = None) -> list[TimeStepResult]:
        """Run the discrete simulation and cache the results."""

        self.results = self.engine.simulate(show_progress=show_progress)
        return self.results

    def ensure_results(self) -> list[TimeStepResult]:
        if self.results is None:
            raise RuntimeError("No simulation data available. Call run() first.")
        return self.results

    def plot_workspace(self, save_path: str | None = None):
        plot_workspace(self.robot, self.config, save_path)

    def plot_results(self, save_path: str | None = None):
        plot_simulation_results(self.ensure_results(), save_path)

    def save_motion_gif(self, gif_path: str | None = None, sample_path: str | None = None):
        create_motion_gif(self.ensure_results(), self.robot, self.config, gif_path, sample_path)


_SIM_INSTANCE: RobotSimulation | None = None


def setup_system(T, dt, link_lengths, A, B, elbows):
    """Backwards compatible helper to configure the default simulation instance."""

    global _SIM_INSTANCE
    config = SimulationConfig(
        T=float(T),
        dt=float(dt),
        link_lengths=list(link_lengths),
        start=A,
        goal=B,
        elbows=tuple(elbows),
    )
    _SIM_INSTANCE = RobotSimulation(config)
    return _SIM_INSTANCE


def _require_sim() -> RobotSimulation:
    global _SIM_INSTANCE
    if _SIM_INSTANCE is None:
        _SIM_INSTANCE = RobotSimulation()
    return _SIM_INSTANCE


def plot_reach(A=None, B=None, save=False):
    sim = _require_sim()
    if A is not None or B is not None:
        cfg = sim.config
        setup_system(
            cfg.T,
            cfg.dt,
            cfg.link_lengths,
            A if A is not None else cfg.start,
            B if B is not None else cfg.goal,
            cfg.elbows,
        )
        sim = _require_sim()
    sim.plot_workspace(_as_path(save, "outputs/workspace.png"))


def simulate():
    sim = _require_sim()
    return sim.run()


def plot_sim_results(results=None, save=False):
    sim = _require_sim()
    data = results if results is not None else sim.ensure_results()
    plot_simulation_results(data, _as_path(save, "outputs/sim_results.png"))


def make_motion_gif(results=None, save=False):
    sim = _require_sim()
    data = results if results is not None else sim.ensure_results()
    create_motion_gif(
        data,
        sim.robot,
        sim.config,
        _as_path(save, "motion gifs/robot_motion.gif"),
        None,
    )


def _as_path(flag_or_path, default_path: str | None):
    if flag_or_path is False or flag_or_path is None:
        return None
    if flag_or_path is True and default_path is not None:
        Path(default_path).parent.mkdir(parents=True, exist_ok=True)
        return default_path
    return flag_or_path


__all__ = [
    "RobotSimulation",
    "SimulationConfig",
    "TimeStepResult",
    "setup_system",
    "simulate",
    "plot_reach",
    "plot_sim_results",
    "make_motion_gif",
]