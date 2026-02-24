# Theoretical Calculations with sympy
# Symbolic derivation of Forward Kinematics, Jacobian, and Trajectory Planning

"""
- Link lengths: l0, l1, l2, l3
- Joint angles: q1, q2, q3
- End-effector position: p_x, p_y, p_z
- Homogeneous transforms: A_g^0, A_0^1, A_1^2, A_2^e, A_g^e
- Jacobian: J(q), J_L(q), J_A(q)
"""

import sympy as sp


# ---------------------------------------------------------------------------
# Basic symbols and DH helper function
# ---------------------------------------------------------------------------

# Link lengths and joint variables
l0, l1, l2, l3 = sp.symbols("l0 l1 l2 l3", real=True)
q1, q2, q3 = sp.symbols("q1 q2 q3", real=True)


def dh(a, d, alpha, theta):
    # Standard DH homogeneous transform 
    ct = sp.cos(theta)
    st = sp.sin(theta)
    ca = sp.cos(alpha)
    sa = sp.sin(alpha)

    return sp.Matrix([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1],
    ])


# ---------------------------------------------------------------------------
# DH parameters and forward kinematics A_g^e
# ---------------------------------------------------------------------------

# DH parameters as used in the report
# i :  a_i   d_i   alpha_i        theta_i
DH_PARAMS = [
    (0,   l0,  sp.pi / 2,       0),     # A_0^g
    (0,   0,  -sp.pi / 2,      q1),     # A_1^0
    (l2,  l1,  0,              q2),     # A_2^1
    (l3,  0,   0,              q3),     # A_e^2
]


def compute_forward_kinematics(VERBOSE=False):
    # Derive forward kinematics from DH parameters
    
    A0g = dh(*DH_PARAMS[0])  # A_0^g
    A1_0 = dh(*DH_PARAMS[1])  # A_1^0
    A2_1 = dh(*DH_PARAMS[2])  # A_2^1
    Ae_2 = dh(*DH_PARAMS[3])  # A_e^2

    # Transforms of all frames with respect to the global frame g
    A1g = sp.simplify(A0g * A1_0)      # A_1^g
    A2g = sp.simplify(A1g * A2_1)      # A_2^g
    Ae_g = sp.simplify(A0g * A1_0 * A2_1 * Ae_2)
    Ae_g = sp.trigsimp(Ae_g)

    if VERBOSE:
        print("=== Forward kinematics A_e^g ===")
        print("A_0^g =")
        sp.pprint(A0g)
        print("\nA_1^0 =")
        sp.pprint(A1_0)
        print("\nA_2^1 =")
        sp.pprint(A2_1)
        print("\nA_e^2 =")
        sp.pprint(Ae_2)

        print("\nA_1^g = A_0^g A_1^0 =")
        sp.pprint(A1g)
        print("\nA_2^g = A_1^g A_2^1 =")
        sp.pprint(A2g)

        print("\nA_e^g = A_0^g A_1^0 A_2^1 A_e^2 =")
        sp.pprint(Ae_g)

    # Return individual DH transforms plus global-frame transforms A_i^g
    return A0g, A1_0, A2_1, Ae_2, Ae_g, A1g, A2g


# ---------------------------------------------------------------------------
# Differential kinematics – geometric Jacobian J(q)
# ---------------------------------------------------------------------------


def compute_jacobian(A0g, A1_0, A2_1, Ae_2, Ae_g, VERBOSE=False):
    # Derive forward differential kinematics (jacobian)
    
    # Cumulative transforms in the global frame
    A1g = sp.simplify(A0g * A1_0)
    A2g = sp.simplify(A1g * A2_1)

    r0 = sp.Matrix([0, 0, 0])
    b0 = sp.Matrix([0, -1, 0])  

    # Joint 1
    r1 = A1g[:3, 3]
    b1 = A1g[:3, 2]

    # Joint 2
    r2 = A2g[:3, 3]
    b2 = A2g[:3, 2]

    # Joint 3
    r3 = Ae_g[:3, 3]
    b3 = Ae_g[:3, 2]

    r_e = r3

    if VERBOSE:
        print("\n=== Intermediate frames in g ===")
        print("A_1^g = A_0^g A_1^0 =")
        sp.pprint(A1g)
        print("\nA_2^g = A_1^g A_2^1 =")
        sp.pprint(A2g)
        print("\nA_e^g =")
        sp.pprint(Ae_g)

        print("\nOrigins r_i in global frame:")
        print("r_1 =")
        sp.pprint(r1)
        print("\nr_2 =")
        sp.pprint(r2)
        print("\nr_3 = r_e =")
        sp.pprint(r3)

        print("\nJoint axes b_i in global frame:")
        print("b_1 =")
        sp.pprint(b1)
        print("\nb_2 =")
        sp.pprint(b2)
        print("\nb_3 =")
        sp.pprint(b3)

    # Linear/angular Jacobian columns
    # J_{L,i} = b_{i-1} x (r_e - r_{i-1}),  J_{A,i} = b_{i-1}

    JL_cols = []
    JA_cols = []

    # i = 1
    JL1 = sp.simplify(b0.cross(r_e - r0))
    JA1 = b0
    JL_cols.append(JL1)
    JA_cols.append(JA1)

    # i = 2
    JL2 = sp.simplify(b1.cross(r_e - r1))
    JA2 = b1
    JL_cols.append(JL2)
    JA_cols.append(JA2)

    # i = 3
    JL3 = sp.simplify(b2.cross(r_e - r2))
    JA3 = b2
    JL_cols.append(JL3)
    JA_cols.append(JA3)

    J_L = sp.Matrix.hstack(*JL_cols)
    J_A = sp.Matrix.hstack(*JA_cols)
    J = sp.Matrix.vstack(J_L, J_A)

    J_L = sp.trigsimp(J_L)
    J_A = sp.trigsimp(J_A)
    J = sp.trigsimp(J)

    if VERBOSE:
        print("\n=== Jacobian columns J_{L_i}, J_{A_i} ===")
        print("J_{L,1} =")
        sp.pprint(JL1)
        print("\nJ_{A,1} =")
        sp.pprint(JA1)
        print("\nJ_{L,2} =")
        sp.pprint(JL2)
        print("\nJ_{A,2} =")
        sp.pprint(JA2)
        print("\nJ_{L,3} =")
        sp.pprint(JL3)
        print("\nJ_{A,3} =")
        sp.pprint(JA3)

        print("\nFull translational Jacobian J_L:")
        sp.pprint(J_L)
        print("\nFull angular Jacobian J_A:")
        sp.pprint(J_A)
        print("\nFull geometric Jacobian J(q):")
        sp.pprint(J)

    return J_L, J_A, J


# ---------------------------------------------------------------------------
# Inverse differential kinematics and singularities
# ---------------------------------------------------------------------------


def compute_inverse_differential(J_L, VERBOSE=False):
    # Singular configs, inverse transform
    
    det_JL = sp.simplify(J_L.det())
    det_JL_factored = sp.factor(det_JL)

    if VERBOSE:
        print("\n=== det(J_L) and singular configurations ===")
        print("det(J_L) =")
        sp.pprint(det_JL)
        print("\nFactored det(J_L) =")
        sp.pprint(det_JL_factored)

    # Adjugate of J_L
    adj_JL = sp.simplify(J_L.adjugate())

    if VERBOSE:
        print("\nAdjugate adj(J_L) used in J_L^{-1} = adj(J_L) / det(J_L):")
        sp.pprint(adj_JL)

    return det_JL_factored, adj_JL



# ---------------------------------------------------------------------------
# Calculate quintic polynomial coefficients
# --------------------------------------------------------------------------- 
def compute_quintic_polynomial(VERBOSE=False):
    # Quintic polynomial s(tau) with via-point at tau = 0.5
    
    tau = sp.symbols("tau", real=True)
    a0, a1, a2, a3, a4, a5 = sp.symbols("a0 a1 a2 a3 a4 a5", real=True)

    s = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3 + a4 * tau**4 + a5 * tau**5
    ds = sp.diff(s, tau)

    eqs = [
        # s(0) = 0 (start at A)
        sp.Eq(s.subs(tau, 0), 0),
        # s(0.5) = 1 (reach B at tau=0.5)
        sp.Eq(s.subs(tau, sp.Rational(1, 2)), 1),
        # s(1) = 0 (return to A)
        sp.Eq(s.subs(tau, 1), 0),
        # v(0) = 0
        sp.Eq(ds.subs(tau, 0), 0),
        # v(0.5) = 0 (stop at B momentarily)
        sp.Eq(ds.subs(tau, sp.Rational(1, 2)), 0),
        # v(1) = 0
        sp.Eq(ds.subs(tau, 1), 0),
    ]

    sol = sp.solve(eqs, (a0, a1, a2, a3, a4, a5), dict=True)[0]

    s_tau = sp.simplify(s.subs(sol))
    ds_tau = sp.simplify(sp.diff(s_tau, tau))

    if VERBOSE:
        print("\n=== Quintic polynomial s(tau) ===")
        print("Solution for coefficients (a0..a5):")
        sp.pprint(sol)
        print("\ns(tau) =")
        sp.pprint(s_tau)
        print("\n\\dot{s}(tau) =")
        sp.pprint(ds_tau)

    return tau, s_tau, ds_tau


# ---------------------------------------------------------------------------
# Main entry point: run all sections once
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VERBOSE = True
    A0g, A1_0, A2_1, Ae_2, Ae_g, A1g, A2g = compute_forward_kinematics()
    J_L, J_A, J = compute_jacobian(A0g, A1_0, A2_1, Ae_2, Ae_g)
    det_JL_factored, adj_JL = compute_inverse_differential(J_L)
    tau, s_tau, ds_tau = compute_quintic_polynomial()