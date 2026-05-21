import numpy as np
import matplotlib.pyplot as plt
from math import comb
from decimal import Decimal, getcontext

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
            dp_table[m+1, j] = half * dp_table[m, j] + half * dp_table[m, (j + 5) % 6]
    return dp_table

def H(r):
    return 1 if 3 <= (r % 6) <= 5 else 0

def calculate_bias(n, a, W, k1, dp_table):
    P00 = Decimal(0)
    P_star0 = Decimal(0)
    min_u = max(0, a - (n - 1 - (W - k1)))
    max_u = min(a, W - k1)
    for u in range(min_u, max_u + 1):
        p_u = nCr_prob(n - 1, a, W - k1, u)
        min_v = max(0, (a // 2) - (a - u))
        max_v = min(a // 2, u)
        for v in range(min_v, max_v + 1):
            p_v_given_u = nCr_prob(a, a // 2, u, v)
            prob_delta = p_u * p_v_given_u
            m_shared = (W - k1) - u
            dp_shared = dp_table[m_shared]
            for j in range(6):
                for x1 in [0, 1]:
                    s_shared = (j + k1 * x1) % 6
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

# Visualizing with Markers
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Bias in symlog scale
ax1.plot(weights, biases, color='teal', label=f'$k_0 = {k0}$', 
         linewidth=1, marker='o', markersize=2, markerfacecolor='black', markeredgewidth=0)

ax1.axhline(0, color='black', lw=1, alpha=0.5)
ax1.set_yscale('symlog', base=2, linthresh=1e-15) 
ax1.set_ylabel(f'$Bias~\\gamma(n, \\alpha, \\beta, k_0)$', fontsize=10)
ax1.set_xlabel('$\\beta  n$', fontsize=10)
ax1.set_title(f'Bias Oscillations $(n={n}, \\alpha n={b})$', fontsize=14)
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend()

# Plot 2: Log of absolute value
# If k0 = 0 then beta * n = n yields a bias of 0 and conversely for k0 = 1
magnitude = np.log2(np.abs(biases))
magnitude_johansson = np.log2(np.abs(johansson_biases))

ax2.plot(weights, magnitude, color='blue', label=f'$\\log_{2}(|\\gamma(n, \\alpha, \\beta, k_0)|), k_0={k0}$', 
         linewidth=1, marker='o', markersize=3, markerfacecolor='black', markeredgewidth=0)
ax2.plot(weights, magnitude_johansson, color='red', linestyle="dashed", label="Johansson et al.")


ax2.set_ylabel('$\\log_{2}(|\\gamma(n, \\alpha, \\beta, k_0)|)$', fontsize=10)
ax2.set_xlabel('$\\beta n$', fontsize=10)
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2.legend()

plt.tight_layout()
plt.show()