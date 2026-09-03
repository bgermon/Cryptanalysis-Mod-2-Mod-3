import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Define Parameters
n = 1000
alpha = 0.2
beta = 0.2

# 2. Map to Hypergeometric parameters
ngood = int(beta * n)     # Marked successes in population
nbad = n - ngood          # Failures in population
nsample = int(alpha * n)  # Total draws

# 3. Define theoretical Normal parameters
mu = alpha * beta * n
variance = n * alpha * beta * (1 - alpha) * (1 - beta)
sigma = np.sqrt(variance)

# 4. Generate 100,000 samples from the hypergeometric distribution
samples = np.random.hypergeometric(ngood, nbad, nsample, size=500000)

# 5. Plotting
plt.figure(figsize=(10, 6))

# Histogram of empirical data
bins = np.arange(min(samples) - 0.5, max(samples) + 1.5, 1)
plt.hist(samples, bins=bins, density=True, alpha=0.6, color='blue', 
         edgecolor='black', label=f'Empirical X')

# Overlay theoretical Normal PDF
x = np.linspace(min(samples), max(samples), 200)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2.5, color='black', linestyle='dashed',
         label=f'Theoretical $\\mathcal{{N}}(\\mu n, \\sigma^2 n)$')

plt.title('Approximation of X by Normal distribution')
plt.xlabel('Value of X')
plt.ylabel('Probability')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()