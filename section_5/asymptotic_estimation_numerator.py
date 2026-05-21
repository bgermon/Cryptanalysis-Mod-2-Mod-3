import sys
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from decimal import Decimal, getcontext
from tqdm import tqdm

# High precision for tiny biases
getcontext().prec = 100

def nCr_prob(pop, draws, succ, k):
    if k < 0 or k > succ or k > draws or (draws - k) > (pop - succ):
        return Decimal(0)
    num = Decimal(comb(succ, k)) * Decimal(comb(pop - succ, draws - k))
    den = Decimal(comb(pop, draws))
    return num / den

def get_all_dp(max_m):
    dp_table = np.zeros((max_m + 1, 6), dtype=object)
    dp_table[0, 0] = Decimal(1)
    for j in range(1, 6):
        dp_table[0, j] = Decimal(0)
    half = Decimal('0.5')
    for m in range(max_m):
        for j in range(6):
            dp_table[m + 1, j] = half * dp_table[m, j] + half * dp_table[m, (j + 5) % 6]
    return dp_table

def H(r):
    return 1 if 3 <= (r % 6) <= 5 else 0

def calculate_exact_numerator(n, a, W, k0, dp_table):
    P11 = Decimal(0)
    min_u = max(0, a - (n - 1 - (W - k0)))
    max_u = min(a, W - k0)
    for u in range(min_u, max_u + 1):
        p_u = nCr_prob(n - 1, a, W - k0, u)
        min_v = max(0, (a // 2) - (a - u))
        max_v = min(a // 2, u)
        for v in range(min_v, max_v + 1):
            p_v_given_u = nCr_prob(a, a // 2, u, v)
            prob_delta = p_u * p_v_given_u
            m_shared = (W - k0) - u
            dp_shared = dp_table[m_shared]
            for j in range(6):
                for x1 in [0, 1]:
                    s_shared = (j + k0 * x1) % 6
                    val_x = (s_shared + v) % 6
                    val_y = (s_shared + u - v) % 6
                    prob_total = prob_delta * dp_shared[j] * Decimal('0.5')
                    if H(val_y) == 0:
                        if H(val_x) == 0:
                            P11 += prob_total
    return P11

PI_SQ = Decimal('9.86960440108935861883449099987615113531369940724079')

def exact_cos_pi_over_6(m):
    m = int(m) % 12
    half = Decimal('0.5')
    sqrt3_over_2 = Decimal('3').sqrt() / Decimal('2')
    
    if m == 0: return Decimal('1')
    elif m in (1, 11): return sqrt3_over_2
    elif m in (2, 10): return half
    elif m in (3, 9): return Decimal('0')
    elif m in (4, 8): return -half
    elif m in (5, 7): return -sqrt3_over_2
    elif m == 6: return Decimal('-1')
    
def asymptotic_numerator(n, alpha, beta, k0):
    n = Decimal(n)
    alpha = Decimal(str(alpha))
    beta = Decimal(str(beta))
    
    b_int = int(round(float(beta * n)))
    k0 = int(k0)
    
    # --- B1 ---
    cos_1a = exact_cos_pi_over_6(b_int - 2)
    cos_1b = exact_cos_pi_over_6(k0)
    B1_amp = (Decimal('2') / Decimal('3')) * cos_1a * cos_1b
    B1_pow = -(PI_SQ * n * beta / Decimal('72')) * (Decimal('1') - alpha * beta)
    B1 = B1_amp * B1_pow.exp()
    
    # --- B2 ---
    cos_2a = exact_cos_pi_over_6(2 * b_int - 4)
    cos_2b = exact_cos_pi_over_6(2 * k0)
    B2_amp = (Decimal('2') / Decimal('9')) * cos_2a * cos_2b
    B2_pow = - (PI_SQ * n * beta / Decimal('18')) * (Decimal('1') - alpha)
    B2 = B2_amp * B2_pow.exp()
    
    # --- B3 ---
    B3_amp = Decimal('2') / Decimal('9')
    B3_pow = -PI_SQ * n * alpha * beta / Decimal('18') * (Decimal('1') - beta)
    B3 = B3_amp * B3_pow.exp()
    
    return Decimal('0.25') + B1 + B2 + B3

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_numerator.py n a")
        exit()
        
    n = int(sys.argv[1])
    a = int(sys.argv[2])
    k0 = 0

    print(f"Testing Numerator for n = {n}, alpha * n = {a}. Please wait...")
    dp_table = get_all_dp(n)
    weights = np.arange(0, n + 1)
    
    exact_biases = []
    approx_biases = []

    for b in tqdm(weights):
        alpha = Decimal(a) / Decimal(n)
        beta = Decimal(int(b)) / Decimal(n)
        
        exact_num = calculate_exact_numerator(n, a, b, k0, dp_table)
        approx_num = asymptotic_numerator(n, alpha, beta, k0)
        
        exact_biases.append(float(exact_num - Decimal('0.25')))
        approx_biases.append(float(approx_num - Decimal('0.25')))

    exact_biases = np.array(exact_biases)
    approx_biases = np.array(approx_biases)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Bias in symlog scale
    ax1.plot(weights, exact_biases, color='teal', label='Exact Bias', 
             linewidth=1, marker='o', markersize=2, markerfacecolor='black', markeredgewidth=0)
    ax1.plot(weights, approx_biases, color='orange', linestyle='--', label='Asymptotic Bias', 
             linewidth=1, marker='o', markersize=2, markerfacecolor='black', markeredgewidth=0)

    ax1.axhline(0, color='black', lw=1, alpha=0.5)
    ax1.set_yscale('symlog', base=2, linthresh=1e-15) 
    ax1.set_ylabel(r'Numerator Bias', fontsize=10)
    ax1.set_xlabel(r'$\beta n$', fontsize=10)
    ax1.set_title(f'Numerator Bias Oscillations $(n={n}, \\alpha n={a}, k_0={k0})$', fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    # Plot 2: Log of absolute value
    if k0 == 0:
        weights_plot = weights[:-1]
        exact_plot = exact_biases[:-1]
        approx_plot = approx_biases[:-1]
    else:
        weights_plot = weights[1:]
        exact_plot = exact_biases[1:]
        approx_plot = approx_biases[1:]

    # Safely compute log2 and replace extreme negative values with NaN
    with np.errstate(divide='ignore'):
        log2_exact = np.log2(np.abs(exact_plot))
        log2_approx = np.log2(np.abs(approx_plot))

    log2_exact[log2_exact < -100] = np.nan
    log2_approx[log2_approx < -100] = np.nan

    ax2.plot(weights_plot, log2_exact, color='teal', label='Exact', 
             linewidth=1, marker='o', markersize=3, markerfacecolor='black', markeredgewidth=0)
    ax2.plot(weights_plot, log2_approx, color='orange', linestyle='--', label='Asymptotic', 
             linewidth=1, marker='o', markersize=3, markerfacecolor='black', markeredgewidth=0)

    ax2.set_ylabel(r'$\log_{2}(|Bias|)$', fontsize=10)
    ax2.set_xlabel(r'$\beta n$', fontsize=10)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.show()