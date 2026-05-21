import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def demonstrate_convergence(n=10000, alpha=0.5, beta=0.5, k0=0, num_samples=500000):
    """
    Simulates the random variables U, V, S and compares the distribution of Z
    to the theoretical Normal distribution.
    """
    a = int(alpha * n)
    b = int(beta * n)
    ell = a // 2

    print(f"Simulating {num_samples} samples for n={n}...\n")

    # 1. Simulate U ~ Hyp(n-1, a, b - k0)
    # Parameters for numpy: ngood=a, nbad=(n-1)-a, nsample=b-k0
    U = np.random.hypergeometric(a, n - 1 - a, b - k0, size=num_samples)

    # 2. Simulate V | U ~ Hyp(a, U, a/2)
    # Vectorized: ngood=U, nbad=a-U, nsample=a/2
    V = np.random.hypergeometric(U, a - U, ell)

    # 3. Simulate S | U ~ Bin(b - k0 - U, 1/2)
    S = np.random.binomial(b - k0 - U, 0.5)

    # 4. Construct Z
    Z = V + S

    # --- Theoretical Moments ---
    theoretical_mean = (b - k0) / 2
    theoretical_var = (beta * (1 - alpha * beta) * n) / 4
    theoretical_std = np.sqrt(theoretical_var)

    # --- Empirical Moments ---
    empirical_mean = np.mean(Z)
    empirical_var = np.var(Z)

    print("--- Mean & Variance Comparison ---")
    print(f"Mean     -> Theoretical: {theoretical_mean:.4f} | Empirical: {empirical_mean:.4f}")
    print(f"Variance -> Theoretical: {theoretical_var:.4f}  | Empirical: {empirical_var:.4f}")
    print("-" * 26)

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    
    # Plot empirical histogram
    bins = np.arange(min(Z) - 0.5, max(Z) + 1.5, 1)
    plt.hist(Z, bins=bins, density=True, alpha=0.6, color='blue', edgecolor='black', label='Empirical (S + V) | U (Histogram)')

    # Plot theoretical Normal PDF
    x_axis = np.linspace(min(Z), max(Z), 1000)
    normal_pdf = norm.pdf(x_axis, theoretical_mean, theoretical_std)
    plt.plot(x_axis, normal_pdf, 'k--', lw=2.5, label=f'Theoretical $\\mathcal{{N}}(\\mu, \\sigma^2)$')

    # Formatting
    plt.title(f'Convergence of (S + V) | U to Normal Distribution (n={n})', fontsize=14, pad=15)
    plt.xlabel('Value of (S + V) | U', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    plt.show()

# Run the simulation
if __name__ == "__main__":
    demonstrate_convergence()