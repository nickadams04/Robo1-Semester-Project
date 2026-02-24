import numpy as np

from scripts.theoretical_calculations import (
    compute_forward_kinematics,
    compute_jacobian,
    compute_inverse_differential,
    compute_quintic_polynomial,
)

from scripts.kinematic_simulation import (
    setup_system,
    plot_reach,
    simulate,
    plot_sim_results,
    make_motion_gif,
)

"""
This script serves as the main entry point for the project.
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
The function `run_simulation` executes the kinematic simulation.
Global variables allow adjustment of system parameters.
It includes initialization via `setup_system()` and execution via `simulate()`.
Visualization options include:

- `plot_reach`: Plots the robot's workspace and target points A & B.
- `plot_results`: Plots position, velocity, joint angles, and angular velocities.
- `make_motion_gif`: Generates a 3D animation of the motion.
"""
def run_simulation():
    # Define constants (movement period, sim period, link lengths, target points)
    T = 3.0
    dt = 1 / 30.0
    link_lengths = [1, 1, 1, 1]
    A = np.array([-1, +1, 2.5])
    B = np.array([+1, -1, 2.5])

    # Define Solution Choice
    # 0: up(+), 1: down(-)
    elbows = (0, 0)

    SAVE = False
    # Initialize all kinematic expressions and numeric parameters
    setup_system(T, dt, link_lengths, A, B, elbows)

    # Plot Robot Workspace limits
    plot_reach(A, B, SAVE)

    # Simulate system
    results = simulate()

    # P, V, q, qdot plots
    plot_sim_results(results, SAVE)

    # GIF and motion samples
    make_motion_gif(results, SAVE)
    
if __name__ == "__main__":
    # run_theoretical_calculations()
    run_simulation()