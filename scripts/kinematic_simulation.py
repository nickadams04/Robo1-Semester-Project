# Kinematic Simulation Module

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import animation
import numpy as np
import sympy as sp
from math import sin, cos, sqrt, atan2, acos, pi
from .theoretical_calculations import (
    compute_forward_kinematics,
    compute_jacobian,
    compute_inverse_differential,
    l0, l1, l2, l3,
    q1, q2, q3,
)
from mpl_toolkits.mplot3d import Axes3D 
from tqdm import tqdm

###############################################################################
# Simulation setup and parameters
###############################################################################

# Define constants (movement period, sim period, link lengths, target points)
T = 3.0   
dt = 1/30.0
link_lengths = [1, 1, 1, 1]
A = np.array([-1, +1, 2.5])
B = np.array([+1, -1, 2.5])

# Define Solution Choice
# 0: up(+), 1: down(-)
elbows = (1, 0)

def setup_system(T_val, dt_val, link_lengths_val, A_val, B_val, elbows_val):
    # Compute and store results from theoretical part
    global T, dt, link_lengths, A, B, elbows
    global A0_g, A1_0, A2_1, Ae_2, Ae_g, A1_g, A2_g
    global J_L, J_A, J, inv_JL, inv_JL_func
    global link_subs
    global offsets, transforms

    # Update module-level simulation parameters from caller
    T = float(T_val)
    dt = float(dt_val)
    link_lengths = list(link_lengths_val)
    A = np.asarray(A_val, dtype=float).ravel()
    B = np.asarray(B_val, dtype=float).ravel()
    elbows = elbows_val

    # Compute kinematics matrices from helper code
    A0_g, A1_0, A2_1, Ae_2, Ae_g, A1_g, A2_g = compute_forward_kinematics()
    J_L, J_A, J = compute_jacobian(A0_g, A1_0, A2_1, Ae_2, Ae_g)
    det_JL_factored, adj_JL = compute_inverse_differential(J_L)
    inv_JL = adj_JL / det_JL_factored

    # Substitute (arbitrary) link lengths using the same symbols as in
    # theoretical_calculations (l0, l1, l2, l3 have real=True there).
    link_subs = {
        l0: link_lengths[0],
        l1: link_lengths[1],
        l2: link_lengths[2],
        l3: link_lengths[3],
    }

    A0_g = A0_g.subs(link_subs)
    A1_0 = A1_0.subs(link_subs)
    A2_1 = A2_1.subs(link_subs)
    Ae_2 = Ae_2.subs(link_subs)
    Ae_g = Ae_g.subs(link_subs)
    A1_g = A1_g.subs(link_subs)
    A2_g = A2_g.subs(link_subs)

    J_L = J_L.subs(link_subs)
    J_A = J_A.subs(link_subs)
    J = J.subs(link_subs)
    inv_JL = inv_JL.subs(link_subs)

    # Numeric inverse Jacobian function for fast evaluation during simulation
    inv_JL_func = sp.lambdify((q1, q2, q3), inv_JL, "numpy")

    # Define joint locations (points of interest for plot) as
    # Transformation + offset
    offsets = [
        sp.Matrix([0, 0, 0]),
        sp.Matrix([0, 0, link_subs[l1]]),
        sp.Matrix([0, 0, 0]),
        sp.Matrix([0, 0, 0]),
    ]
    transforms = [A0_g, A1_g, A2_g, Ae_g]

def plot_reach(A = None, B=None, save = False):
    # Plot the reach of the robot to determine valid points
    # Sample joint space and compute workspace
    px_expr, py_expr, pz_expr = Ae_g[0, 3], Ae_g[1, 3], Ae_g[2, 3]
    fk_num = sp.lambdify((q1, q2, q3), (px_expr, py_expr, pz_expr), "numpy")

    q1_vals = np.linspace(-np.pi, np.pi, 50)
    q2_vals = np.linspace(-np.pi, np.pi, 25)
    q3_vals = np.linspace(-np.pi, np.pi, 25)

    Q1, Q2, Q3 = np.meshgrid(q1_vals, q2_vals, q3_vals, indexing="ij")
    X, Y, Z = fk_num(Q1, Q2, Q3)
    x = np.asarray(X).ravel()
    y = np.asarray(Y).ravel()
    z = np.asarray(Z).ravel()

    # Joint positions for q = 0
    angle_subs_zero = {q1: 0, q2: 0, q3: 0}
    joint_positions = [[0, 0, 0]]
    for T, offset in zip(transforms, offsets):
        T_num = sp.Matrix(T.subs(angle_subs_zero))
        p_h = T_num * sp.Matrix.vstack(offset, sp.Matrix([1]))
        joint_positions.append(np.array(p_h[:3], dtype=float).flatten())
    joint_positions = np.vstack(joint_positions)

    # Plots: x-z and x-y views
    fig, (ax_xz, ax_xy) = plt.subplots(1, 2, figsize=(10, 5))

    # x-z view
    ax_xz.scatter(x, z, s=1, alpha=0.3)
    ax_xz.plot(joint_positions[:, 0], joint_positions[:, 2], "r-o", label="q=0")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title("Workspace projection (x-z)")
    ax_xz.axis("equal")

    # x-y view
    ax_xy.scatter(x, y, s=1, alpha=0.3)
    ax_xy.plot(joint_positions[:, 0], joint_positions[:, 1], "r-o", label="q=0")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("Workspace projection (x-y)")
    ax_xy.axis("equal")

    # Plot A and B if provided
    if A is not None and B is not None:
        A = np.asarray(A, dtype=float).ravel()
        B = np.asarray(B, dtype=float).ravel()

        # x-z view (use x and z components)
        ax_xz.plot([A[0], B[0]], [A[2], B[2]], "g-o", label="A-B")
        
        # x-y view (use x and y components)
        ax_xy.plot([A[0], B[0]], [A[1], B[1]], "g-o", label="A-B")

    ax_xz.legend()
    ax_xy.legend()
    
    if save:
        fig.savefig("latex_code/images/reach_workspace.pdf", format="pdf", bbox_inches="tight")
    
    plt.tight_layout()
    plt.show()

def get_parameterized(tau):
    # Compute the parameterized position and velocity for normalized time tau
    # s = tau**2 * (3 - 2*tau)
    # s_dot = 6*tau*(1-tau)
    s = 16 * tau**2 * (tau-1)**2
    s_dot = 32 * tau * (2 * tau**2 -3 * tau + 1)
    P_k = A + s * (B - A)
    V_k = s_dot * (B - A) / T
    
    return P_k, V_k


def inverse_kinematics(P):
    # Compute inverse kinematics of end-effector position P
    # signs determines the solution selection
    px, py, pz = P
    l0, l1, l2, l3 = link_lengths
    
    r = sqrt(px**2 + (pz - l0)**2)
    phi = atan2(-px, pz - l0)
    q1 = phi + ((-1) ** elbows[0]) * acos(l1 / r)
    
    R = px * cos(q1) + (pz-l0) * sin(q1)
    c3 = (R**2 + py**2 -l2**2-l3**2) / (2*l2*l3)
    s3 = ((-1) ** elbows[1]) * sqrt(1-c3**2)
    
    q3 = atan2(s3, c3)
    q2 = atan2(py, R) - atan2(l3*s3, l2 + l3*c3)
    return q1, q2, q3
    
class TimeStepResult:
    def __init__(self, t, tau, p, v, q, qdot):
        self.t = float(t)
        self.tau = float(tau)
        self.p = np.asarray(p, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.q = np.asarray(q, dtype=float)
        self.qdot = np.asarray(qdot, dtype=float)
        
def control_loop(t):
    # Calculate normalized time
    tau = t / T
    
    # Calculate smoothstep position and velocity
    P_k, V_k = get_parameterized(tau)
    
    # Calculate q frok IK (double elbow-up)
    q1_sol, q2_sol, q3_sol = inverse_kinematics(P_k)
    
    # Calculate tool velocity from differential FK
    inv_JL_val = np.array(inv_JL_func(q1_sol, q2_sol, q3_sol), dtype=float)
    
    # Calculate joint angular velocities
    q1_dot, q2_dot, q3_dot = inv_JL_val @ V_k
    
    return TimeStepResult(
        t = t, tau = tau,
        p = P_k,
        v = V_k,
        q = [q1_sol, q2_sol, q3_sol],
        qdot = [q1_dot, q2_dot, q3_dot]
    )
    
def simulate():
    times = np.arange(0, T+dt, dt)
    results = []

    for t in tqdm(times, desc="Simulating", unit="step"):
        results.append(control_loop(t))

    return results

def plot_sim_results(results, save = False):
    ts = np.array([r.t for r in results])
    p = np.array([r.p for r in results])
    v = np.array([r.v for r in results])
    q = np.array([r.q for r in results])
    qdot = np.array([r.qdot for r in results])

    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex="col")

    # Row 1: p
    labels_p = [r"$p_x$", r"$p_y$", r"$p_z$"]
    for j in range(3):
        axes[0, j].plot(ts, p[:, j])
        axes[0, j].set_ylabel(labels_p[j])
        axes[0, j].set_title(labels_p[j] + " vs $t$")
        axes[0, j].grid(True)

    # Row 2: v
    labels_v = [r"$v_x$", r"$v_y$", r"$v_z$"]
    for j in range(3):
        axes[1, j].plot(ts, v[:, j])
        axes[1, j].set_ylabel(labels_v[j])
        axes[1, j].set_title(labels_v[j] + " vs $t$")
        axes[1, j].grid(True)

    # Row 3: q
    labels_q = [r"$q_1 (rad)$", r"$q_2 (rad)$", r"$q_3 (rad)$"]
    tick_vals = [-1, -0.5, 0, 0.5, 1]
    tick_positions = pi * np.array(tick_vals)
    tick_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$",r"$\pi$"]
    for j in range(3):
        axes[2, j].plot(ts, q[:, j])
        axes[2, j].set_ylabel(labels_q[j])
        axes[2, j].set_yticks(tick_positions)
        axes[2, j].set_yticklabels(tick_labels)
        axes[2, j].set_title(labels_q[j] + " vs $t$")
        axes[2, j].grid(True)

    # Row 4: qdot
    labels_qd = [r"$\dot{q}_1 (rad/s)$", r"$\dot{q}_2 (rad/s)$", r"$\dot{q}_3$ (rad/s)"]
    for j in range(3):
        axes[3, j].plot(ts, qdot[:, j])
        axes[3, j].set_ylabel(labels_qd[j])
        axes[3, j].set_yticks(tick_positions)
        axes[3, j].set_yticklabels(tick_labels)
        axes[3, j].set_title(labels_qd[j] + " vs $t$")
        axes[3, j].set_xlabel(r"$t$")
        axes[3, j].grid(True)

    fig.tight_layout()
    
    if save:
        fig.savefig("latex_code/images/sim_results_neg.pdf", format="pdf", bbox_inches="tight")
    
    plt.show()

def compute_joint_positions(q_vals):
    # Helper: compute joint positions in 3D for given joint angles
    q1_val, q2_val, q3_val = q_vals
    angle_subs = {q1: q1_val, q2: q2_val, q3: q3_val}

    joint_positions = [[0, 0, 0]]
    for T, offset in zip(transforms, offsets):
        T_num = sp.Matrix(T.subs(angle_subs))
        p_h = T_num * sp.Matrix.vstack(offset, sp.Matrix([1]))
        joint_positions.append(np.array(p_h[:3], dtype=float).flatten())

    return np.vstack(joint_positions)

def make_motion_gif(results, save = False):
    # Create a 3D GIF of the robot movement over time
    joint_trajs = [compute_joint_positions(r.q) for r in results]

    # Determine axis limits from all frames
    all_points = np.vstack(joint_trajs)
    xs, ys, zs = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min())
    mid_x = 0.5 * (xs.max() + xs.min())
    mid_y = 0.5 * (ys.max() + ys.min())
    mid_z = 0.5 * (zs.max() + zs.min())

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot([], [], [], "bo-", lw=2)

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Robot movement in 3D")

    # Plot target points A and B and their connecting line
    ax.scatter(A[0], A[1], A[2], c="r", marker="o", label="A")
    ax.scatter(B[0], B[1], B[2], c="g", marker="o", label="B")
    ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], "k--", label="A-B")
    ax.legend()

    def update(frame_idx):
        pts = joint_trajs[frame_idx]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        return line,

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(joint_trajs),
        interval=1000 * dt,
        blit=False,
    )

    fps = int(1.0 / dt)
    if save:
        anim.save("latex_code/images/robot_motion.gif", writer="pillow", fps=fps)

    # Sample 4 configurations and
    # plot them together in a single 3D figure, along with A, B,
    # and the connecting segment A-B.
    sample_taus = [0.0, 0.3, 0.5, 0.8]
    n_frames = len(joint_trajs)
    sample_indices = [int(tau * (n_frames - 1)) for tau in sample_taus]
    colors = ["r", "g", "b", "m"]

    fig_samples = plt.figure(figsize=(6, 6))
    ax_samples = fig_samples.add_subplot(111, projection="3d")

    for tau_val, idx, color in zip(sample_taus, sample_indices, colors):
        pts = joint_trajs[idx]
        ax_samples.plot(
            pts[:, 0], pts[:, 1], pts[:, 2],
            marker="o", color=color,
            label=f"$\\tau = {tau_val:.1f}$",
        )

    ax_samples.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax_samples.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax_samples.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    ax_samples.set_xlabel("x")
    ax_samples.set_ylabel("y")
    ax_samples.set_zlabel("z")
    ax_samples.set_title("Robot configurations at $\\tau=0,0.3,0.5,0.8$")

    # Plot A, B and A-B on the same figure
    ax_samples.scatter(A[0], A[1], A[2], c="r", marker="o", label="A")
    ax_samples.scatter(B[0], B[1], B[2], c="g", marker="o", label="B")
    ax_samples.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], "k--", label="A-B")
    ax_samples.legend()
    if save:
        fig_samples.savefig(
            "latex_code/images/robot_motion_samples_neg.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    # Show the GIF animation window as well (not only save it)
    plt.show()
    
if __name__ == "__main__":
    SAVE = False
    # Initialize all kinematic expressions and numeric parameters
    setup_system(T, dt, link_lengths, A, B, elbows)

    # Plot Robot Workspace limits
    # plot_reach(A, B, SAVE)
    
    # Simulate system
    results = simulate()
    
    # P, V, q, qdot plots
    # plot_sim_results(results, SAVE)
    
    # GIF and motion samples
    make_motion_gif(results, SAVE)