import math
from decimal import Decimal, getcontext

# Set precision to 300 digits
getcontext().prec = 300

def compute_real_residue(alpha, beta, n):
    """Computes exact hypergeometric bias using pure integer arithmetic."""
    betan = int(beta * n)
    alphan = int(alpha * n)
    
    # Calculate exact mathematical bounds for j
    j_min = max(0, alphan - (n - betan))
    j_max = min(alphan, betan)
    
    ctr = 0
    for j in range(j_min, j_max + 1):
        if j % 6 in (3, 4, 5):
            ctr += math.comb(betan, j) * math.comb(n - betan, alphan - j)
            
    total_comb = math.comb(n, alphan)
    
    return (Decimal(2 * ctr - total_comb) / Decimal(2 * total_comb))


n = 384
# Generate exact Decimal fractions: 0.1, 0.3, 0.5, 0.7, 0.9
fractions = [Decimal(i) / Decimal(10) for i in range(1, 10, 2)]

for alpha in fractions:
    for beta in fractions:
        tmp = compute_real_residue(alpha, beta, n)
        
        is_positive = tmp > 0
        abs_tmp = abs(tmp)
        
        if abs_tmp == 0:
            log2_val = float('-inf')
        else:
            log2_val = abs_tmp.ln() / Decimal('2').ln()
            
        print(f"{alpha:.1f} {beta:.1f} {is_positive} {log2_val:.3f}")
        
    print()