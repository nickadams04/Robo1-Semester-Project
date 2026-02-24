from scripts.theoretical_calculations import (
    compute_forward_kinematics,
    compute_jacobian,
    compute_inverse_differential,
    compute_quintic_polynomial,
)

from scripts.kinematic_simulation import RobotSimulation, SimulationConfig

"""
This script exposes helper functions for symbolic derivations and simulations.
"""

"""
The function `run_theoretical_calculations` performs symbolic computations
such as the composition of DH matrices, Jacobian calculation and its inverse,
as well as the solution of the 5th-degree polynomial trajectory generation.
"""
def run_theoretical_calculations():
    VERBOSE = True
    A0g, A1_0, A2_1, Ae_2, Ae_g, A1g, A2g = compute_forward_kinematics(VERBOSE)
    J_L, J_A, J = compute_jacobian(A0g, A1_0, A2_1, Ae_2, Ae_g, VERBOSE)
    det_JL_factored, adj_JL = compute_inverse_differential(J_L, VERBOSE)
    tau, s_tau, ds_tau = compute_quintic_polynomial(VERBOSE)

"""
Run the high-level simulation pipeline (workspace plot, batch sim, GIF export).
"""
def run_simulation():
    config = SimulationConfig()
    simulation = RobotSimulation(config)

    simulation.plot_workspace()
    results = simulation.run()
    simulation.plot_results()
    simulation.save_motion_gif("motion gifs/robot_motion_demo.gif")

    return results
    
if __name__ == "__main__":
    # run_theoretical_calculations()
    run_simulation()