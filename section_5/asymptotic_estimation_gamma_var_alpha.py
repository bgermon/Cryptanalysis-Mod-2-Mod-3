import numpy as np
import matplotlib.pyplot as plt
from math import comb
from decimal import Decimal, getcontext

# High precision for tiny biases
getcontext().prec = 200

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
            dp_table[m+1, j] = half * dp_table[m, j] + half * dp_table[m, (j + 5) % 6]
    return dp_table

def H(r):
    return 1 if 3 <= (r % 6) <= 5 else 0

def calculate_bias(n, a, b, k0, dp_table):
    P00 = Decimal(0)
    P_star0 = Decimal(0)
    min_u = max(0, a - (n - 1 - (b - k0)))
    max_u = min(a, b - k0)
    for u in range(min_u, max_u + 1):
        p_u = nCr_prob(n - 1, a, b - k0, u)
        min_v = max(0, (a // 2) - (a - u))
        max_v = min(a // 2, u)
        for v in range(min_v, max_v + 1):
            p_v_given_u = nCr_prob(a, a // 2, u, v)
            prob_delta = p_u * p_v_given_u
            m_shared = (b - k0) - u
            dp_shared = dp_table[m_shared]
            for j in range(6):
                for x1 in [0, 1]:
                    s_shared = (j + k0 * x1) % 6
                    val_x = (s_shared + v) % 6
                    val_y = (s_shared + u - v) % 6
                    prob_total = prob_delta * dp_shared[j] * Decimal('0.5')
                    if H(val_y) == 0:
                        P_star0 += prob_total
                        if H(val_x) == 0:
                            P00 += prob_total
    if P_star0== 0: 
        return 0
    return float((P00 / P_star0) - Decimal(0.5))

PI_SQ = Decimal('9.86960440108935861883449099987615113531369940724079')

def exact_cos_pi_over_6(m):
    """
    Calcule exactement la valeur algébrique de cos(m * pi / 6) en Decimal,
    où m est un entier. Cela remplace les np.cos() flottants.
    """
    m = int(m) % 12
    half = Decimal('0.5')
    sqrt3_over_2 = Decimal('3').sqrt() / Decimal('2')
    
    if m == 0:
        return Decimal('1')
    elif m in (1, 11):
        return sqrt3_over_2
    elif m in (2, 10):
        return half
    elif m in (3, 9):
        return Decimal('0')
    elif m in (4, 8):
        return -half
    elif m in (5, 7):
        return -sqrt3_over_2
    elif m == 6:
        return Decimal('-1')
    
def asymptotic_numerator(n, alpha, beta, k0):
    """
    Computes the asymptotic approximation using purely Decimal operations.
    """
    n = Decimal(n)
    alpha = Decimal(alpha)
    beta = Decimal(beta)
    
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
    
    # Total approximated probability
    P_approx = Decimal('0.25') + B1 + B2 + B3
    
    return P_approx

def full_asymptotic_expansion(n, alpha, beta, k0):
    """
    Computes the complete 16-term asymptotic limit using the Decimal module.
    """
    
    # Convert inputs to Decimal
    d_alpha = Decimal(str(alpha))
    d_beta = Decimal(str(beta))
    d_n = Decimal(n)
    
    # be assume 'b' must be an integer (b = beta * n)
    b_int = int(round(d_beta * n))
    
    # Frequency indices
    indices = [-1, 0, 1, 3]
    
    # Amplitudes |c_j|
    abs_c = {
        0: Decimal('1') / Decimal('2'),
        1: Decimal('1') / Decimal('3'),
        -1: Decimal('1') / Decimal('3'),
        3: Decimal('1') / Decimal('6')
    }
    
    # Phase shift multipliers A_j (in units of pi/6)
    A = {
        0: 0,
        1: -2,
        -1: 2,
        3: 0
    }
    
    total_prob = Decimal('0')
    
    # Precompute common factors for the decay exponent
    factor1 = (Decimal('1') - alpha)
    factor2 = alpha * (Decimal('1') - beta)
    prefactor = -(PI_SQ * d_n * beta) / Decimal('72')
    
    # Loop over all 16 modes
    for j in indices:
        for k in indices:
            # Calculate integer multipliers for the phase (modulo 12 logic)
            M = (j + k) * b_int + A[j] + A[k]
            N = (j + k) * k0
            
            # Exact Amplitude
            amp = abs_c[j] * abs_c[k] * exact_cos_pi_over_6(M) * exact_cos_pi_over_6(N)
            
            # If amplitude is exactly zero, skip expensive exp calculation
            if amp == Decimal('0'):
                continue
                
            # Gaussian Decay Exponent
            decay_val = prefactor * (factor1 * Decimal((j + k)**2) + factor2 * Decimal((j - k)**2))
            
            # Full Term
            term = amp * decay_val.exp()
            total_prob += term
            
    return total_prob

def asymptotic_denominator(n, alpha, beta, k0):
    """
    Computes the asymptotic bias using the 100-digit precision Decimal library.
    n: integer
    alpha: float or string
    beta: float or string
    k0: integer (0 or 1)
    """
    # Convert inputs to Decimal
    d_alpha = Decimal(str(alpha))
    d_beta = Decimal(str(beta))
    d_n = Decimal(n)
    
    # Calculate b = beta * n (assumed to be an integer based on the setup)
    b_int = int(d_beta * d_n)
    
    cos_1a = exact_cos_pi_over_6(b_int - 2)
    cos_1b = exact_cos_pi_over_6(k0)

    C1_amp = (Decimal('2') / Decimal('3')) * cos_1a * cos_1b
    C1_pow = -(PI_SQ * d_n * beta / Decimal('72')) * (Decimal('1') - d_alpha * d_beta)

    C1 = C1_amp * C1_pow.exp()
    
    # Total asymptotic bias
    return Decimal('0.5') + C1

def asymptotic_bias(n, a, b, k0):
    n_dec = Decimal(n)
    a_dec = Decimal(int(a))
    b_dec = Decimal(int(b))
    
    alpha = a_dec / n_dec
    beta = b_dec / n_dec

    num = full_asymptotic_expansion(n_dec, alpha, beta, k0)
    den = asymptotic_denominator(n, alpha, beta, k0)

    if den == Decimal('0'):
        return Decimal('0')

    prob = num / den
    return float(prob - Decimal('0.5'))


import sys

if len(sys.argv) < 3:
    print("Usage: python script.py n beta * n")
    exit()
else:
    n = int(sys.argv[1])
    b = int(sys.argv[2])


print(f"Calculating for n={n}, beta * n ={b}. Please wait...")
dp_table = get_all_dp(n)
weights = np.arange(0, n + 1, 2)
from tqdm import tqdm

k0 = 0
biases = np.array([calculate_bias(n, a, b, k0, dp_table) for a in tqdm(weights)])
johansson_biases = np.array([2**-(0.41*(a // 2) + 1.17) for a in weights])
asymp_biases = np.array([asymptotic_bias(n, a, b, k0) for a in tqdm(weights)])
valid = asymp_biases != 0
abs_ratios = np.abs(biases[valid] / asymp_biases[valid])

max_ratio = np.max(abs_ratios)
mean_ratio = np.mean(abs_ratios)

print(f"Maximum absolute ratio: {max_ratio}")
print(f"Mean absolute ratio: {mean_ratio}")

# Visualizing with Markers
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Bias in symlog scale
ax1.plot(weights, biases, color='teal', label=f'$k_0 = {k0}$', 
         linewidth=1, marker='o', markersize=2, markerfacecolor='black', markeredgewidth=0)

ax1.plot(weights, asymp_biases, color='orange', label=f'$k_0 = {k0}$', 
         linewidth=1, marker='o', markersize=2, markerfacecolor='black', markeredgewidth=0)

ax1.axhline(0, color='black', lw=1, alpha=0.5)
ax1.set_yscale('symlog', base=2, linthresh=1e-15) 
ax1.set_ylabel(f'$Bias~\\gamma(n, \\alpha, \\beta, k_0)$', fontsize=10)
ax1.set_xlabel('$\\beta  n$', fontsize=10)
ax1.set_title(f'Bias Oscillations $(n={n}, \\beta n={b})$', fontsize=14)
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend()

# Plot 2: Log of absolute value
# If k0 = 0 then beta * n = n yields a bias of 0 and conversely for k0 = 1
magnitude = np.log2(np.abs(biases))
magnitude_johansson = np.log2(np.abs(johansson_biases))
magnitude_asymp = np.log2(np.abs(asymp_biases))

ax2.plot(weights, magnitude, color='blue', label=f'$\\log_{2}(|\\gamma(n, \\alpha, \\beta, k_0)|), k_0={k0}$', 
         linewidth=1, marker='o', markersize=3, markerfacecolor='black', markeredgewidth=0)
ax2.plot(weights, magnitude_johansson, color='red', linestyle="dashed", label="Johansson et al.")
ax2.plot(weights, magnitude_asymp, color='orange', label=f'$Asymptotic$', 
         linewidth=1, marker='o', markersize=3, markerfacecolor='black', markeredgewidth=0)


ax2.set_ylabel('$\\log_{2}(|\\gamma(n, \\alpha, \\beta, k_0)|)$', fontsize=10)
ax2.set_xlabel('$\\beta n$', fontsize=10)
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2.legend()

plt.tight_layout()
plt.show()