import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def run_asymptotic_experiment(n=5000, alpha=0.25, beta=0.5, k0=1, num_samples=100000000):
    """
    Simulates the exact discrete generation of X and Y and compares them 
    to their theoretical asymptotic Gaussian limits.
    """
    print(f"Running simulation with {num_samples:,} samples for n={n}...\n")
    
    # Structural parameters
    a = int(alpha * n)
    b = int(beta * n)
    l = a // 2
    
    # 1. Draw U ~ Hypergeometric(pop=n-1, successes=a, draws=b-k0)
    U = np.random.hypergeometric(ngood=a, nbad=n - 1 - a, nsample=b - k0, size=num_samples)
    
    # 2. Draw V|U ~ Hypergeometric(pop=a, successes=U, draws=l)
    V = np.random.hypergeometric(ngood=U, nbad=a - U, nsample=l, size=num_samples)
    
    # 3. Draw S|U ~ Binomial(trials=b-k0-U, p=0.5)
    S = np.random.binomial(n=b - k0 - U, p=0.5, size=num_samples)
    
    # 4. Draw x0 ~ Bernoulli(0.5)
    x0 = np.random.binomial(n=1, p=0.5, size=num_samples)
    
    # Construct the diagonalized variables
    Y = 2 * k0 * x0 + 2 * S + U
    Z = 2 * V - U
    
    # --- Theoretical Asymptotic Limits ---
    # Mean of X accounts for the Bernoulli mixture E[x0] = 0.5
    mu_Y_theory = b
    var_Y_theory = n * beta * (1 - alpha)
    std_Y_theory = np.sqrt(var_Y_theory)
    
    mu_Z_theory = 0
    var_Z_theory = n * alpha * beta * (1 - beta)
    std_Z_theory = np.sqrt(var_Z_theory)
    
    # --- Print Statistics ---
    print("--- Empirical vs Theoretical Statistics ---")
    print(f"Mean(Y): Empirical = {np.mean(Y):.3f} | Theory = {mu_Y_theory:.3f}")
    print(f"Var(Y):  Empirical = {np.var(Y):.3f} | Theory = {var_Y_theory:.3f}")
    print(f"Mean(Z): Empirical = {np.mean(Z):.3f} | Theory = {mu_Z_theory:.3f}")
    print(f"Var(Z):  Empirical = {np.var(Z):.3f} | Theory = {var_Z_theory:.3f}")
    
    corr = np.corrcoef(Y, Z)[0, 1]
    print(f"\nCorrelation(Y, Z): {corr:.5f}")
    
    # --- Plotting ---
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Histogram of X
    ax0 = plt.subplot(1, 3, 1)
    ax0.hist(Y, bins=50, density=True, alpha=0.6, color='blue', label='Empirical Y')
    x_axis = np.linspace(mu_Y_theory - 4*std_Y_theory, mu_Y_theory + 4*std_Y_theory, 200)
    ax0.plot(x_axis, norm.pdf(x_axis, mu_Y_theory, std_Y_theory), 'k--', lw=2, label=r'Theory $N(\mu_y, \sigma_Y²)$')
    ax0.set_title("Marginal Distribution of Y")
    ax0.legend()
    
    # Plot 2: Histogram of Y
    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(Z, bins=50, density=True, alpha=0.6, color='green', label='Empirical Z')
    y_axis = np.linspace(mu_Z_theory - 4*std_Z_theory, mu_Z_theory + 4*std_Z_theory, 200)
    ax2.plot(y_axis, norm.pdf(y_axis, mu_Z_theory, std_Z_theory), 'k--', lw=2, label=r'Theory $N(0, \sigma_Z²)$')
    ax2.set_title("Marginal Distribution of Z")
    ax2.legend()
    
    # Plot 3: 2D Histogram (Joint Distribution)
    ax3 = plt.subplot(1, 3, 3)
    h = ax3.hist2d(Y, Z, bins=40, cmap='magma', density=True)
    ax3.set_title("Joint Distribution (Y, Z)")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    fig.colorbar(h[3], ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_asymptotic_experiment()