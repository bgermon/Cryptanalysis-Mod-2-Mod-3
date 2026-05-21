import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# --- High Precision Configuration ---
getcontext().prec = 100

# --- Cached Constants for Speed ---
DECIMAL_0 = Decimal('0')
DECIMAL_1 = Decimal('1')
DECIMAL_2 = Decimal('2')
DECIMAL_0_5 = Decimal('0.5')
LOG2_OF_2 = DECIMAL_2.ln()

# Global cache for log2 factorials (populated per-process in init_worker)
LOG2_FACT_CACHE = []

def d_log2(x):
    """High precision log2 using Decimal."""
    if x <= DECIMAL_0: return Decimal('-Infinity')
    return x.ln() / LOG2_OF_2

def init_worker(max_n):
    """Initializes globals and caches factorials for each parallel worker process."""
    getcontext().prec = 100
    global LOG2_FACT_CACHE
    LOG2_FACT_CACHE = [DECIMAL_0] * (max_n + 1)
    current = DECIMAL_0
    for i in range(1, max_n + 1):
        current += d_log2(Decimal(i))
        LOG2_FACT_CACHE[i] = current

def log2_binom(n, k):
    """O(1) calculation using precomputed log2 factorials."""
    if k < 0 or k > n: return Decimal('-Infinity')
    if k == 0 or k == n: return DECIMAL_0
    return LOG2_FACT_CACHE[n] - LOG2_FACT_CACHE[k] - LOG2_FACT_CACHE[n - k]

def H(p):
    """Binary entropy function H(p)."""
    if p <= DECIMAL_0 or p >= DECIMAL_1: return DECIMAL_0
    return -p * d_log2(p) - (DECIMAL_1 - p) * d_log2(DECIMAL_1 - p)

def H_inv(y):
    """Inverse binary entropy H^-1(y) using binary search."""
    if y <= DECIMAL_0: return DECIMAL_0
    if y >= DECIMAL_1: return DECIMAL_0_5
    
    low = DECIMAL_0
    high = DECIMAL_0_5
    for _ in range(50):  
        mid = (low + high) / DECIMAL_2
        if H(mid) < y: low = mid
        else: high = mid
    return mid

def get_complexity_a(n_val, k, alpha, log2_bias):
    term_n = log2_binom(n_val, k)
    term_an = log2_binom(k, k // 2)
    
    c = (DECIMAL_2 * -log2_bias + DECIMAL_1 - term_n - term_an + Decimal(n_val) * (DECIMAL_1 + alpha)) / DECIMAL_2
    lambd = c / Decimal(n_val)

    sym_alpha = alpha if alpha <= DECIMAL_0_5 else DECIMAL_1 - alpha

    target_h = DECIMAL_1 - lambd
    if target_h < DECIMAL_0: target_h = DECIMAL_0
    elif target_h > DECIMAL_1: target_h = DECIMAL_1
    
    h_inv_val = H_inv(target_h)
    
    denom = DECIMAL_1 - sym_alpha
    if denom == DECIMAL_0:
        p_prime = DECIMAL_0
    else:
        p_prime = (h_inv_val - (sym_alpha / DECIMAL_2)) / denom
        
    if p_prime < DECIMAL_0: p_prime = DECIMAL_0
    elif p_prime > DECIMAL_1: p_prime = DECIMAL_1
    
    y = denom * (DECIMAL_1 - H(p_prime))
        
    e1 = y * Decimal(n_val)
    e2 = -DECIMAL_2 * log2_bias 
    
    # --- LOG-SUM-EXP TRICK ---
    if e1 > e2:
        diff = e2 - e1
        total_log2_dec = e1 + d_log2(DECIMAL_1 + DECIMAL_2**diff)
    else:
        diff = e1 - e2
        total_log2_dec = e2 + d_log2(DECIMAL_1 + DECIMAL_2**diff)
        
    return total_log2_dec, c

def process_n_task(args):
    """Task function run by worker processes."""
    n_val, bias_data = args
    
    min_complexity_all = Decimal('Infinity')
    min_complexity_skip = Decimal('Infinity')
    best_alpha_all = None
    best_alpha_skip = None
    best_data_all = None
    best_data_skip = None
    
    closest_k_037 = int(0.37 * n_val)
    if closest_k_037 % 2 != 0: closest_k_037 += 1
    closest_k_037 = min(n_val, max(0, closest_k_037))
    
    complexity_037 = None

    for k, bias_all_str, bias_skip_str in bias_data:
        alpha = Decimal(k) / Decimal(n_val)
        
        # Calculate for "All"
        if bias_all_str != '0.0':
            bias_all = Decimal(bias_all_str)
            log2_bias_all = d_log2(bias_all)
            total_log2_all, c_all = get_complexity_a(n_val, k, alpha, log2_bias_all)
            
            if total_log2_all < min_complexity_all:
                min_complexity_all = total_log2_all
                best_alpha_all = alpha
                best_data_all = c_all

        # Calculate for "Skip"
        if bias_skip_str != '0.0':
            bias_skip = Decimal(bias_skip_str)
            log2_bias_skip = d_log2(bias_skip)
            total_log2_skip, c_skip = get_complexity_a(n_val, k, alpha, log2_bias_skip)
            
            if total_log2_skip < min_complexity_skip:
                min_complexity_skip = total_log2_skip
                best_alpha_skip = alpha
                best_data_skip = c_skip

        # Calculate for Johansson
        if k == closest_k_037:
            johansson_bias = -(Decimal('0.41') * alpha * Decimal(n_val) * DECIMAL_0_5 + Decimal('1.17'))
            complexity_037, data_037 = get_complexity_a(n_val, k, alpha, johansson_bias)


    return (
        n_val, 
        float(best_alpha_all) if best_alpha_all is not None else None, 
        float(best_data_all) if best_data_all is not None else None,
        float(min_complexity_all) if min_complexity_all != Decimal('Infinity') else None, 
        float(best_alpha_skip) if best_alpha_skip is not None else None, 
        float(best_data_skip) if best_data_skip is not None else None,
        float(min_complexity_skip) if min_complexity_skip != Decimal('Infinity') else None, 
        float(complexity_037) if complexity_037 is not None else None,
        float(data_037) if data_037 is not None else None
    )

if __name__ == '__main__':
    # --- Main Execution ---
    try:
        df = pd.read_csv('asymptotic_complexity.csv')
    except FileNotFoundError:
        print("Could not find 'asymptotic_complexity.csv'. Please run the C++ executable first.")
        sys.exit(1)

    max_n_in_dataset = int(df['n'].max())
    print("Pre-processing dataset for multiprocessing...")
    
    tasks = []
    # Convert biases to strings to prevent float precision loss before passing to Decimal
    df['abs_bias_all_str'] = df['asymp_bias_all'].abs().astype(str)
    df['abs_bias_skip_str'] = df['asymp_bias_skip_5_mod_6'].abs().astype(str)
    
    for n_val, group in df.groupby('n'):
        if n_val >= max_n_in_dataset:
            continue
        # Filter out evens, but keep row if EITHER bias is non-zero
        valid_rows = group[(group['a'] % 2 == 0) & ((group['abs_bias_all_str'] != '0.0') | (group['abs_bias_skip_str'] != '0.0'))]
        
        # Package data as list of tuples: (k, bias_all, bias_skip)
        bias_data = list(zip(valid_rows['a'].astype(int), valid_rows['abs_bias_all_str'], valid_rows['abs_bias_skip_str']))
        
        if bias_data:
            tasks.append((int(n_val), bias_data))

    num_cores = 48
    print(f"Launching processing across {num_cores} parallel workers...")

    results = []
    with ProcessPoolExecutor(max_workers=num_cores, initializer=init_worker, initargs=(max_n_in_dataset,)) as executor:
        futures = {executor.submit(process_n_task, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Complexities"):
            results.append(future.result())

    # Sort results
    results.sort(key=lambda x: x[0])

    plot_n = []
    plot_min_c_all, plot_min_c_skip, plot_c_037, plot_best_data_all, plot_best_data_skip, plot_data_037 = [], [], [], [], [], []
    arr_a_all, arr_a_skip = [], []
    for res in results:
        n, best_alpha_all, best_data_all, min_c_all, best_alpha_skip, best_data_skip, min_c_skip, c_037, data_037 = res
        # Only append 'n' if we got at least one valid complexity
        if min_c_all is not None or min_c_skip is not None:
            plot_n.append(n)
            plot_min_c_all.append(min_c_all)
            plot_min_c_skip.append(min_c_skip)
            plot_c_037.append(c_037)
            plot_best_data_all.append(best_data_all)
            plot_best_data_skip.append(best_data_skip)
            plot_data_037.append(data_037)
            arr_a_all.append(best_alpha_all)
            arr_a_skip.append(best_alpha_skip)
            
    print(f"\nProcessed {len(plot_n)} different 'n' values.")
    # --- Plotting Time Complexity ---
    plt.figure(figsize=(10, 6))

    # Filter out None values dynamically for plotting
    valid_n_all = [n for i, n in enumerate(plot_n) if plot_min_c_all[i] is not None]
    valid_c_all = [c for c in plot_min_c_all if c is not None]

    valid_n_skip = [n for i, n in enumerate(plot_n) if plot_min_c_skip[i] is not None]
    valid_c_skip = [c for c in plot_min_c_skip if c is not None]

    valid_n_037 = [n for i, n in enumerate(plot_n) if plot_c_037[i] is not None]
    valid_c_037 = [c for c in plot_c_037 if c is not None]

    if valid_n_all:
        plt.plot(valid_n_all, valid_c_all, label='Ours (All keys)', color='blue', linewidth=1)
        
    if valid_n_skip:
        plt.plot(valid_n_skip, valid_c_skip, label=r'Ours ($h \neq 5$ mod $6$)', color='green', linewidth=1)

    if valid_n_037:
        plt.plot(valid_n_037, valid_c_037, label='Johansson et al.', color='red', linestyle='dashed')

    plt.xlabel('Parameter $n$')
    plt.ylabel(r'Time Complexity Exponent ($\log_2$)')
    plt.title('Asymptotic Complexity Comparison')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.savefig("figure_asymp_comp.png", format="png", dpi=300, bbox_inches='tight')
    print("\nPlot saved successfully as 'figure_asymp_comp.png'.")
    # --- Calculate Slopes ---
    slope_all = np.polyfit(valid_n_all, valid_c_all, 1)[0] if len(valid_n_all) > 1 else 0
    slope_skip = np.polyfit(valid_n_skip, valid_c_skip, 1)[0] if len(valid_n_skip) > 1 else 0
    slope_037 = np.polyfit(valid_n_037, valid_c_037, 1)[0] if len(valid_n_037) > 1 else 0

    # --- Plotting Data Complexity ---
    plt.figure(figsize=(10, 6))

    # Filter out None values dynamically for plotting
    valid_n_all = [n for i, n in enumerate(plot_n) if plot_best_data_all[i] is not None]
    valid_data_all = [c for c in plot_best_data_all if c is not None]

    valid_n_skip = [n for i, n in enumerate(plot_n) if plot_best_data_skip[i] is not None]
    valid_data_skip = [c for c in plot_best_data_skip if c is not None]

    valid_n_037 = [n for i, n in enumerate(plot_n) if plot_data_037[i] is not None]
    valid_data_037 = [c for c in plot_data_037 if c is not None]

    if valid_n_all:
        plt.plot(valid_n_all, valid_data_all, label='Ours (All keys)', color='blue', linewidth=1)
        
    if valid_n_skip:
        plt.plot(valid_n_skip, valid_data_skip, label=r'Ours ($h \neq 5$ mod $6$)', color='green', linewidth=1)

    if valid_n_037:
        plt.plot(valid_n_037, valid_data_037, label='Johansson et al.', color='red', linestyle='dashed')

    plt.xlabel('Parameter $n$')
    plt.ylabel(r'Data Complexity Exponent ($\log_2$)')
    plt.title('Asymptotic Complexity Comparison')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # --- Calculate Slopes ---
    slope_all = np.polyfit(valid_n_all, valid_data_all, 1)[0] if len(valid_n_all) > 1 else 0
    slope_skip = np.polyfit(valid_n_skip, valid_data_skip, 1)[0] if len(valid_n_skip) > 1 else 0
    slope_037 = np.polyfit(valid_n_037, valid_data_037, 1)[0] if len(valid_n_037) > 1 else 0

    # Print to the console
    print("\nCalculated Slopes:")
    print(f"Ours (All Constraints):  {slope_all:.5f}")
    print(f"Ours (Skip b ≢ 5 mod 6): {slope_skip:.5f}")
    print(f"Johansson et al.:        {slope_037:.5f}")