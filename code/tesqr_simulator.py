
import qutip as qt
from qutip.qip.operations import cnot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Enable LaTeX rendering for plots (optional)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# --- Plotting Function ---
def plot_3d_density_matrix(ax, rho):
    """
    Plot a 3D bar graph of the real part of a density matrix with enhanced visualization.
    """
    if rho is None:
        ax.text(0.5, 0.5, 0, 'No data', horizontalalignment='center', verticalalignment='center')
        return

    mat = rho.full().real
    dim = mat.shape[0]

    if dim == 4:
        basis_labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    else:
        basis_labels = [f"|{i}>" for i in range(dim)]

    ax.set_xticks(np.arange(dim))
    ax.set_xticklabels(basis_labels, fontsize=12)
    ax.set_yticks(np.arange(dim))
    ax.set_yticklabels(basis_labels, fontsize=12)
    
    _x, _y = np.meshgrid(np.arange(dim), np.arange(dim))
    x = _x.flatten()
    y = _y.flatten()
    z = np.zeros_like(x, dtype=float)
    dx = dy = 0.8
    values = mat.flatten()
    
    vmax = np.max(np.abs(values))
    
    colors = []
    for val in values:
        if val < 0:
            colors.append(plt.cm.Reds(-val/vmax))
        else:
            colors.append(plt.cm.Blues(val/vmax))
    
    ax.bar3d(x, y, z, dx, dy, values, shade=True, color=colors)
    ax.set_xlabel("Row (Basis)", fontsize=12)
    ax.set_ylabel("Column (Basis)", fontsize=12)
    ax.set_zlabel("Real Part", fontsize=12)
    ax.set_zlim(-vmax, vmax)

# --- Helper Functions ---
def apply_kraus_to_state_vector(psi, kraus_ops):
    """
    Apply one Kraus operator to state vector probabilistically.
    """
    probs = [qt.expect(K.dag() * K, psi) for K in kraus_ops]
    total_prob = sum(probs)
    if total_prob < 1e-10:
        return psi
    probs = [p / total_prob for p in probs]
    idx = np.random.choice(len(kraus_ops), p=probs)
    psi_out = kraus_ops[idx] * psi
    return psi_out.unit()

def sample_pure_state(rho):
    """
    Sample a pure state from a density matrix by eigen-decomposition.
    """
    eigvals, eigvecs = rho.eigenstates()
    probs = [max(0, ev.real) for ev in eigvals]
    total = sum(probs)
    if total < 1e-10:
        return eigvecs[0]
    probs = [p / total for p in probs]
    idx = np.random.choice(len(eigvals), p=probs)
    return eigvecs[idx]

# --- Loss and Transduction Channels ---
def generate_loss_kraus_ops(eta, dim_photonic):
    """Kraus ops for a pure-loss channel on a truncated Fock space."""
    kraus_ops = []
    for k in range(dim_photonic):
        K = qt.Qobj(np.zeros((dim_photonic, dim_photonic)))
        for n in range(k, dim_photonic):
            coeff = math.sqrt(math.comb(n, k)) * (eta ** ((n - k) / 2)) * ((1 - eta) ** (k / 2))
            K += coeff * qt.basis(dim_photonic, n - k) * qt.basis(dim_photonic, n).dag()
        kraus_ops.append(K)
    return kraus_ops

def apply_fiber_loss_state_vector(psi, eta_c, dim_photonic):
    """Apply fiber channel loss to a state vector probabilistically."""
    kraus = generate_loss_kraus_ops(eta_c, dim_photonic)
    ops = [qt.tensor(K, qt.qeye(2)) for K in kraus]
    return apply_kraus_to_state_vector(psi, ops)

def apply_loss_channel(rho, eta, rho_env, mode_index, total_modes):
    """Gaussian-loss coupling between system mode and thermal environment mode."""
    dim_ph = rho.dims[0][0]
    dim_aux = rho_env.dims[0][0]
    a = qt.destroy(dim_ph)
    b = qt.destroy(dim_aux)
    I_ph = qt.qeye(dim_ph)
    I_q = qt.qeye(2)
    I_aux = qt.qeye(dim_aux)

    # build full tensor: [photonic modes..., qubit B, environment]
    ops = [I_ph] * total_modes + [I_q] + [I_aux]
    a_op = qt.tensor(*(ops[:mode_index] + [a] + ops[mode_index+1:]))
    b_op = qt.tensor(*([I_ph] * total_modes + [I_q, b]))

    theta = np.arccos(np.sqrt(eta))
    U = (theta * (a_op.dag() * b_op - a_op * b_op.dag())).expm()
    rho_tot = U * qt.tensor(rho, rho_env) * U.dag()
    return rho_tot.ptrace(list(range(total_modes + 1)))

def simulate_model2_partial_probabilistic(P, eta_c, eta_t, n_bar, dim_photonic):
    """
    Source generation, fiber loss, thermal transduction, then sample pure outcome.
    """
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    p0 = qt.basis(dim_photonic, 0)
    p1 = qt.basis(dim_photonic, 1)

    # 1) Source generation
    if np.random.rand() < P:
        psi = (qt.tensor(p0, g) + qt.tensor(p1, e)) / np.sqrt(2)
    else:
        psi = qt.tensor(p0, g)

    # 2) Fiber loss
    psi = apply_fiber_loss_state_vector(psi, eta_c, dim_photonic)
    rho = psi * psi.dag()

    # 3) Transduction
    rho_env = qt.thermal_dm(dim_photonic, n_bar)
    rho = apply_loss_channel(rho, eta_t, rho_env, mode_index=0, total_modes=1)

    # 4) Sample pure state
    return sample_pure_state(rho)

# --- Coupling to Matter Qubit ---
def apply_coupling_state_vector(psi_A_B, eta_qb, eta_ph, dim_photonic):
    """Couple photonic mode A and qubit B into matter qubit M probabilistically."""
    g = qt.basis(2, 0)
    psi = qt.tensor(psi_A_B, g)
    eta = eta_qb * eta_ph

    # Kraus elements
    K0 = (qt.tensor(g * g.dag(), qt.basis(dim_photonic, 0) * qt.basis(dim_photonic, 0).dag())
          - 1j * np.sqrt(eta) * qt.tensor(qt.basis(2, 1) * qt.basis(2, 0).dag(), qt.basis(dim_photonic, 0) * qt.basis(dim_photonic, 1).dag()))
    K1 = np.sqrt(1 - eta) * qt.tensor(qt.basis(2, 0) * qt.basis(2, 0).dag(), qt.basis(dim_photonic, 0) * qt.basis(dim_photonic, 1).dag())
    I_B = qt.qeye(2)

    # Permute to order A ⊗ B ⊗ M
    K0_full = qt.tensor(K0, I_B).permute([1, 2, 0])
    K1_full = qt.tensor(K1, I_B).permute([1, 2, 0])

    return apply_kraus_to_state_vector(psi, [K0_full, K1_full])

# --- Imperfect Gates and Noise ---
def get_depolarizing_kraus(p):
    """Generate Kraus operators for depolarizing channel."""
    I = qt.qeye(2)
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    return [np.sqrt(1 - p) * I, np.sqrt(p / 3) * X, np.sqrt(p / 3) * Y, np.sqrt(p / 3) * Z]

# --- New Functions for Multiple Purification Rounds ---
def apply_U_rot_with_noise(rho_BM, p_s):
    """Apply U_rot with depolarizing noise on M."""
    S = qt.Qobj([[1, 0], [0, 1j]])
    U_rot = qt.tensor(qt.qeye(2), S)
    rho_rot = U_rot * rho_BM * U_rot.dag()
    kraus_M = get_depolarizing_kraus(p_s)
    rho_rot = sum([qt.tensor(qt.qeye(2), K) * rho_rot * qt.tensor(qt.qeye(2), K).dag() for K in kraus_M])
    return rho_rot

def apply_U_rot_dag_with_noise(rho_BM_rot, p_s):
    """Apply U_rot.dag() with depolarizing noise on M."""
    S = qt.Qobj([[1, 0], [0, 1j]])
    U_rot_dag = qt.tensor(qt.qeye(2), S.dag())
    rho = U_rot_dag * rho_BM_rot * U_rot_dag.dag()
    kraus_M = get_depolarizing_kraus(p_s)
    rho = sum([qt.tensor(qt.qeye(2), K) * rho * qt.tensor(qt.qeye(2), K).dag() for K in kraus_M])
    return rho

def purify_pair_density(rho_BM1_rot, rho_BM2_rot, p_cnot, epsilon):
    """Purify two BM density matrices in the rotated basis, returning purified density matrix if successful."""
    rho_total = qt.tensor(rho_BM1_rot, rho_BM2_rot)  # [B1, M1, B2, M2]

    # Apply noisy CNOT B1->B2 (indices 0->2)
    CNOT_B1_B2 = cnot(N=4, control=0, target=2)
    rho_total = CNOT_B1_B2 * rho_total * CNOT_B1_B2.dag()
    kraus = get_depolarizing_kraus(p_cnot)
    rho_total = sum([qt.tensor(K, qt.qeye(2), qt.qeye(2), qt.qeye(2)) * rho_total * qt.tensor(K.dag(), qt.qeye(2), qt.qeye(2), qt.qeye(2)).dag() for K in kraus])
    rho_total = sum([qt.tensor(qt.qeye(2), qt.qeye(2), K, qt.qeye(2)) * rho_total * qt.tensor(qt.qeye(2), qt.qeye(2), K.dag(), qt.qeye(2)).dag() for K in kraus])

    # Apply noisy CNOT M1->M2 (indices 1->3)
    CNOT_M1_M2 = cnot(N=4, control=1, target=3)
    rho_total = CNOT_M1_M2 * rho_total * CNOT_M1_M2.dag()
    rho_total = sum([qt.tensor(qt.qeye(2), K, qt.qeye(2), qt.qeye(2)) * rho_total * qt.tensor(qt.qeye(2), K.dag(), qt.qeye(2), qt.qeye(2)).dag() for K in kraus])
    rho_total = sum([qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), K) * rho_total * qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), K.dag()).dag() for K in kraus])

    # Measure B2 (index 2)
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    M_g = np.sqrt(1 - epsilon) * g * g.dag() + np.sqrt(epsilon) * e * e.dag()
    M_e = np.sqrt(epsilon) * g * g.dag() + np.sqrt(1 - epsilon) * e * e.dag()
    M_g_B2 = qt.tensor(qt.qeye(2), qt.qeye(2), M_g, qt.qeye(2))
    M_e_B2 = qt.tensor(qt.qeye(2), qt.qeye(2), M_e, qt.qeye(2))
    p_g_B2 = (M_g_B2 * rho_total * M_g_B2.dag()).tr()
    p_e_B2 = (M_e_B2 * rho_total * M_e_B2.dag()).tr()
    total_p = p_g_B2 + p_e_B2
    if total_p < 1e-10:
        return None, False
    p_g_B2 /= total_p
    outcome_B2 = 'g' if np.random.rand() < p_g_B2 else 'e'
    rho_total = (M_g_B2 * rho_total * M_g_B2.dag()) / p_g_B2 if outcome_B2 == 'g' else (M_e_B2 * rho_total * M_e_B2.dag()) / p_e_B2

    # Measure M2 (index 3)
    M_g_M2 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), M_g)
    M_e_M2 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), M_e)
    p_g_M2 = (M_g_M2 * rho_total * M_g_M2.dag()).tr()
    p_e_M2 = (M_e_M2 * rho_total * M_e_M2.dag()).tr()
    total_p = p_g_M2 + p_e_M2
    if total_p < 1e-10:
        return None, False
    p_g_M2 /= total_p
    outcome_M2 = 'g' if np.random.rand() < p_g_M2 else 'e'
    rho_total = (M_g_M2 * rho_total * M_g_M2.dag()) / p_g_M2 if outcome_M2 == 'g' else (M_e_M2 * rho_total * M_e_M2.dag()) / p_e_M2

    if outcome_B2 == outcome_M2:
        rho_B1M1_rot = rho_total.ptrace([0, 1])
        return rho_B1M1_rot, True
    return None, False

def perform_purification_rounds(n, params):
    """Perform n rounds of purification, returning final rho_BM_rot if all succeed."""
    rho_list = []
    for _ in range(2**n):
        psi = simulate_model2_partial_probabilistic(params['P'], params['eta_c'], params['eta_t'], params['n_bar'],params['dim_photonic'])
        psi = apply_coupling_state_vector(psi, params['eta_qb'], params['eta_ph'],params['dim_photonic'])
        rho_BM = psi.ptrace([1, 2])
        rho_BM_rot = apply_U_rot_with_noise(rho_BM, params['p_s'])
        rho_list.append(rho_BM_rot)

    for _ in range(n):
        new_rho_list = []
        for i in range(0, len(rho_list), 2):
            rho_p, ok = purify_pair_density(rho_list[i], rho_list[i + 1], params['p_cnot'], params['epsilon'])
            if not ok:
                return None
            new_rho_list.append(rho_p)
        rho_list = new_rho_list

    return rho_list[0] if len(rho_list) == 1 else None

def imperfect_swap_and_measure_density(rho_4q, p_cnot, p_h, p_t, epsilon):
    """Perform swap and measurement on 4-qubit density matrix."""
    # CNOT M1->M2 (1->3)
    CNOT_M1_M2 = cnot(N=4, control=1, target=3)
    rho_4q = CNOT_M1_M2 * rho_4q * CNOT_M1_M2.dag()
    kraus = get_depolarizing_kraus(p_cnot)
    rho_4q = sum([qt.tensor(qt.qeye(2), K, qt.qeye(2), qt.qeye(2)) * rho_4q * qt.tensor(qt.qeye(2), K.dag(), qt.qeye(2), qt.qeye(2)).dag() for K in kraus])
    rho_4q = sum([qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), K) * rho_4q * qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), K.dag()).dag() for K in kraus])

    # Hadamard on M1 (index 1)
    H = (1 / np.sqrt(2)) * qt.Qobj([[1, 1], [1, -1]])
    U_H = qt.tensor(qt.qeye(2), H, qt.qeye(2), qt.qeye(2))
    rho_4q = U_H * rho_4q * U_H.dag()
    kraus_H = get_depolarizing_kraus(p_h)
    rho_4q = sum([qt.tensor(qt.qeye(2), K, qt.qeye(2), qt.qeye(2)) * rho_4q * qt.tensor(qt.qeye(2), K.dag(), qt.qeye(2), qt.qeye(2)).dag() for K in kraus_H])

    # Depolarizing decoherence on B1 and B2
    for idx in [0, 2]:
        kraus_D = get_depolarizing_kraus(p_t)
        ops_D = [qt.tensor(*[K if i == idx else qt.qeye(2) for i in range(4)]) for K in kraus_D]
        rho_4q = sum([op * rho_4q * op.dag() for op in ops_D])

    # Measure M1 (index 1)
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    M_g = np.sqrt(1 - epsilon) * g * g.dag() + np.sqrt(epsilon) * e * e.dag()
    M_e = np.sqrt(epsilon) * g * g.dag() + np.sqrt(1 - epsilon) * e * e.dag()
    M_g1 = qt.tensor(qt.qeye(2), M_g, qt.qeye(2), qt.qeye(2))
    M_e1 = qt.tensor(qt.qeye(2), M_e, qt.qeye(2), qt.qeye(2))
    p_g1 = (M_g1 * rho_4q * M_g1.dag()).tr()
    p_e1 = (M_e1 * rho_4q * M_e1.dag()).tr()
    total_p1 = p_g1 + p_e1
    if total_p1 < 1e-10:
        return None, None
    p_g1 /= total_p1
    outcome1 = 'g' if np.random.rand() < p_g1 else 'e'
    rho_4q = (M_g1 * rho_4q * M_g1.dag()) / p_g1 if outcome1 == 'g' else (M_e1 * rho_4q * M_e1.dag()) / p_e1

    # Measure M2 (index 3)
    M_g2 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), M_g)
    M_e2 = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), M_e)
    p_g2 = (M_g2 * rho_4q * M_g2.dag()).tr()
    p_e2 = (M_e2 * rho_4q * M_e2.dag()).tr()
    total_p2 = p_g2 + p_e2
    if total_p2 < 1e-10:
        return None, None
    p_g2 /= total_p2
    outcome2 = 'g' if np.random.rand() < p_g2 else 'e'
    rho_4q = (M_g2 * rho_4q * M_g2.dag()) / p_g2 if outcome2 == 'g' else (M_e2 * rho_4q * M_e2.dag()) / p_e2

    rho_B1B2 = rho_4q.ptrace([0, 2])
    return (outcome1, outcome2), rho_B1B2

# --- Main Simulation with Multiple Purification Rounds ---
def extended_sim_with_purification_n_rounds(params, n_rounds, num_trials):
    """
    Runs n rounds of purification on each side, then swap+measurement, collecting average fidelities and density matrices.
    """
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    ideal_states = {
        ('g', 'g'): (1 / np.sqrt(2)) * (qt.tensor(g, g) - qt.tensor(e, e)),
        ('e', 'g'): (1 / np.sqrt(2)) * (qt.tensor(g, g) + qt.tensor(e, e)),
        ('g', 'e'): (-1j / np.sqrt(2)) * (qt.tensor(g, e) + qt.tensor(e, g)),
        ('e', 'e'): (1j / np.sqrt(2)) * (-qt.tensor(g, e) + qt.tensor(e, g))
    }
    outcomes = list(ideal_states.keys())
    results = {o: {'counts': 0, 'fidelity_sum': 0, 'rho_sum': qt.Qobj(np.zeros((4,4)), dims=[[2,2],[2,2]])} for o in outcomes}

    for _ in range(num_trials):
        # Side 1: n rounds of purification
        rho_BM1_rot = perform_purification_rounds(n_rounds, params)
        if rho_BM1_rot is None:
            continue
        rho_BM1 = apply_U_rot_dag_with_noise(rho_BM1_rot, params['p_s'])

        # Side 2: n rounds of purification
        rho_BM2_rot = perform_purification_rounds(n_rounds, params)
        if rho_BM2_rot is None:
            continue
        rho_BM2 = apply_U_rot_dag_with_noise(rho_BM2_rot, params['p_s'])

        # Assemble 4-qubit state [B1, M1, B2, M2]
        rho_4q = qt.tensor(rho_BM1, rho_BM2)

        # Swap and measurement
        outcome, rho_B1B2 = imperfect_swap_and_measure_density(
            rho_4q,
            params['p_cnot'], params['p_h'], params['p_t'], params['epsilon']
        )
        if outcome is None:
            continue

        # Compute fidelity
        F = qt.fidelity(rho_B1B2, ideal_states[outcome]) ** 2
        results[outcome]['counts'] += 1
        results[outcome]['fidelity_sum'] += F
        results[outcome]['rho_sum'] += rho_B1B2

    # Compute averages
    final = {}
    for o in outcomes:
        c = results[o]['counts']
        final[o] = {
            'probability': c / num_trials,
            'fidelity': results[o]['fidelity_sum'] / c if c > 0 else 0,
            'avg_rho': results[o]['rho_sum'] / c if c > 0 else None
        }
    return final

# --- Example Usage ---
if __name__ == '__main__':
    params = {
        'P': 1.0,
        'eta_c': 0.4029,
        'eta_t': 1,
        'n_bar': 0.01,
        'eta_qb': 0.99,
        'eta_ph': 0.99,
        'p_cnot': 0.002,
        'p_h': 0.0002,
        'p_t': 0,
        'epsilon': 0.005,
        'p_s': 0.0002,
        'dim_photonic':2
    }
    n_rounds = 1  # Example: 3 rounds of purification
    results = extended_sim_with_purification_n_rounds(params, n_rounds, num_trials=20000)
    for outcome, data in results.items():
        print(f"\nOutcome {outcome}:")
        print(f"  Probability = {data['probability']:.4f}")
        print(f"  Fidelity    = {data['fidelity']:.4f}")

        # Se esiste la matrice densità media, la stampiamo
        if data['avg_rho'] is not None:
            print("  Average density matrix (ρ̄_B1B2):")
            print(data['avg_rho'])
        else:
            print("  No density matrix available for this outcome.")


    # Plot the average density matrices
    outcomes = [('g', 'g'), ('g', 'e'), ('e', 'g'), ('e', 'e')]
    fig = plt.figure(figsize=(12, 10))
    for i, o in enumerate(outcomes):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        avg_rho = results[o]['avg_rho']
        plot_3d_density_matrix(ax, avg_rho)
        ax.set_title(f"M1={o[0]}, M2={o[1]}")
    plt.tight_layout()
    plt.savefig('density_matrices.png')
