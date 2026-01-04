# Νικόλαος Αδαμόπουλος
# 03122074

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
Μέσω αυτού του αρχείου μπορείτε να εκτελέσετε τον κώδικα που χρησιμοποιήθηκε στα δύο μέρη της άσκησης.
"""

"""
Η συνάρτηση `run_theoretical_calculations` υπολογίζει και τυπώνει τους διάφορους υπολογισμούς που 
παραλείφθηκαν στην αναφορά, όπως η σύνθεση των πινάκων της μεθόδου DH, ο υπολογισμός της ιακωβιανής 
και της αντίστροφής της καθώς και η επίλυση του πολυωνύμου 5ου βαθμού στο δεύτερο μέρος.
"""
def run_theoretical_calculations():
    VERBOSE = True
    A0g, A1_0, A2_1, Ae_2, Ae_g, A1g, A2g = compute_forward_kinematics(VERBOSE)
    J_L, J_A, J = compute_jacobian(A0g, A1_0, A2_1, Ae_2, Ae_g, VERBOSE)
    det_JL_factored, adj_JL = compute_inverse_differential(J_L, VERBOSE)
    tau, s_tau, ds_tau = compute_quintic_polynomial(VERBOSE)

"""
Η συνάρτηση `run_simulation` τρέχει την απαιτούμενη προσομοίωση για το μέρος Β.
Μέσω των global μεταβλητών μπορούμε να προσαρμόσουμε μεταβλητές που θεωρούνται
δεδομένες στους υπολογισμούς.
Πέραν των `setup_system()` και `simulate()` που κάνουν τις απαραίτητες αρχικοποιήσεις
και προσομοιώσεις αντίστοιχα, μπορούμε να επιλέξουμε από τρείς συναρτήσεις με γραφικά
αποτελέσματα

- `plot_reach`: Πλοτάρει την περιοχή εργασίας του βραχίονα και τα επιλεγμένα σημεία
                Α και Β. Χρησιμο σε περίπτωση που θέλουμε να αλλάξουμε τα Α, Β.

- `plot_results`: Πλοτάρει τα διαγράμματα των ζητούμενων μεγεθών.

- `make_motion_gif`: Δημιουργεί το 3-διάστατο animation ολόκληρης της κίνησης καθώς
                     και ζητούμενο το διάγραμμα με τα δείγματα της κίνησης.
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