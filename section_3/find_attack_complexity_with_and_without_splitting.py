#This script was created by Claude Opus 4.7 and corrected and verified by the authors


#!/usr/bin/env python3
"""
Complexity estimation for the distinguishing attacks against the
alternating mod-2/mod-3 weak PRF.

This script estimates:
  * Section 3.2 -- the *aggregated* distinguishing attack.
        Input : n (key/input length), w_k (Hamming weight of the key, = beta*n).
        Output: N, the number of data (queries) required for the attack to work.

  * Section 4.3 -- the *aggregated splitting strategy*.
        Input : n, s (split size), w_k (Hamming weight of the key on part J).
        Output: data complexity and time complexity.

Design notes
------------
* Everything is kept modular: each "key quantity" (the biases epsilon and eta,
  the weight probabilities, the binomials, ...) lives in its own small function.
* Readability is preferred over raw performance.
* All numerical work is done with mpmath at high precision, because the biases
  are *extremely* small (e.g. 2^-40) while the binomial / 2^n factors are
  *extremely* large.  Doing this in plain floats would underflow/overflow.
  Final magnitudes are reported as base-2 logarithms (log2 of the complexity),
  which is the natural unit for cryptanalytic complexities.
"""

from mpmath import mp, mpf, binomial as _mp_binomial, log, sqrt

# High working precision. The biases can be of the order 2^-400 in the
# asymptotic regime, so we need plenty of decimal digits to stay accurate.
mp.dps = 100


# ---------------------------------------------------------------------------
# Precomputation: a memoized binomial coefficient.
# ---------------------------------------------------------------------------
# Every bias / probability formula in this file is built out of binomial
# coefficients C(n, k), and the same coefficients are recomputed thousands of
# times -- in particular while sweeping the split size s.  We therefore cache
# them.  The cache stores full-precision mpf values, so this is a pure speed-up
# with NO loss of precision (the numbers returned are bit-identical to calling
# mpmath.binomial directly).
#
# Out-of-range coefficients (k < 0 or k > n) are 0 by convention, which is
# exactly what the combinatorial sums need.
_binom_cache = {}


def binomial(n, k):
    """Memoized C(n, k) as an mpf. Returns 0 for k < 0 or k > n."""
    n = int(n)
    k = int(k)
    if k < 0 or k > n:
        return mpf(0)
    key = (n, k)
    val = _binom_cache.get(key)
    if val is None:
        val = _mp_binomial(n, k)
        _binom_cache[key] = val
    return val


def clear_caches():
    """Reset all precomputation caches (useful between independent runs)."""
    _binom_cache.clear()
    _pow2_cache.clear()


# Powers of two 2^m appear as the size of the input space; cache them too.
_pow2_cache = {}


def pow2(m):
    """Memoized 2^m as an mpf."""
    m = int(m)
    val = _pow2_cache.get(m)
    if val is None:
        val = mpf(2) ** m
        _pow2_cache[m] = val
    return val


# ---------------------------------------------------------------------------
# 0. Small shared helpers
# ---------------------------------------------------------------------------

# The function H : Z -> F_2 of the construction (Section 2.2):
#   H(w) = 0 if (w mod 6) in {0, 1, 2}
#   H(w) = 1 otherwise (i.e. (w mod 6) in {3, 4, 5}).
_H_ONE_RESIDUES = (3, 4, 5)


def H(w):
    """The Boolean function H of the mod-2/mod-3 construction."""
    return 1 if (w % 6) in _H_ONE_RESIDUES else 0


def log2(x):
    """Base-2 logarithm of a (positive) mpmath number, returned as a float."""
    return float(log(mpf(x), 2))


def weight_probability(w, n):
    """
    Probability p_{w} that a uniformly random x in F_2^n has Hamming weight w:
        p_w = C(n, w) / 2^n.
    (Denoted p_{w/n} in the paper.)
    """
    return binomial(n, w) / pow2(n)


# ---------------------------------------------------------------------------
# 1. The bias epsilon(alpha, beta, n)  --  Lemma 1, Equation (1)
# ---------------------------------------------------------------------------
#
#   Pr_x[ F(k, x) = 1 | wt(x) = alpha*n ] = 1/2 + epsilon(alpha, beta, n),
#
# where, with an = alpha*n and bn = beta*n,
#
#   epsilon = C(n, an)^{-1} * sum_{i : i mod 6 in {3,4,5}} C(bn, i) * C(n-bn, an-i)
#             - 1/2.
#
# We parametrise directly by the integer weights (w_x for the input, w_k for the
# key) since that is what the attacks actually manipulate.

def epsilon_bias(w_x, w_k, n):
    """
    Exact bias epsilon for input Hamming weight w_x and key Hamming weight w_k,
    over inputs/keys of length n.  Returns an mpf (can be negative).
    """
    numerator = mpf(0)
    for i in range(0, w_k + 1):
        if i % 6 in _H_ONE_RESIDUES:
            numerator += binomial(w_k, i) * binomial(n - w_k, w_x - i)
    return numerator / binomial(n, w_x) - mpf(1) / 2


# ---------------------------------------------------------------------------
# 2. The bias eta_{(alpha, beta, n-s)}(m)  --  Lemma 3
# ---------------------------------------------------------------------------
#
# For any m in Z_6 and a fixed xJ of weight alpha*(n-s):
#
#   Pr[ H(m + <kJ, xJ>) = 1 | wt(kJ) = beta*(n-s) ] = 1/2 + eta(m),
#
# where, with nm = n - s, an = alpha*nm and bn = beta*nm,
#
#   eta(m) = -1/2 + C(nm, bn)^{-1} * sum_{j=0}^{nm} H(m + j) * C(an, j) * C(nm-an, bn-j).
#
# Here we again parametrise by integer weights:
#   w_xJ : Hamming weight of xJ (the "alpha*(n-s)" part)
#   w_kJ : Hamming weight of kJ (the "beta*(n-s)" part)
#   nm   : length of the J part, i.e. n - s.

def eta_bias(m, w_xJ, w_kJ, nm):
    """
    Exact bias eta_{(.,.,nm)}(m) for residue m in Z_6, input weight w_xJ and key
    weight w_kJ on the part J of length nm = n - s.  Returns an mpf.
    """
    numerator = mpf(0)
    for j in range(0, nm + 1):
        if H(m + j):  # only the residues {3,4,5} contribute
            numerator += binomial(w_xJ, j) * binomial(nm - w_xJ, w_kJ - j)
    return numerator / binomial(nm, w_kJ) - mpf(1) / 2


def eta_max_squared(w_xJ, w_kJ, nm):
    """
    max_{m in Z_6} eta_{(.,.,nm)}(m)^2  -- the quantity that drives the splitting
    strategy's data complexity (see Theorem 1 / Theorem 2).

    Implementation note: the per-index term
        T(j) = C(w_xJ, j) * C(nm - w_xJ, w_kJ - j)
    does not depend on the residue m -- only on which j are kept (those with
    H(m + j) = 1).  We therefore compute the T(j) once and, for each residue m,
    sum the terms with H(m + j) = 1.  This is ~6x faster than calling eta_bias
    six times and gives identical results.
    """
    # T(j) is non-zero only when 0 <= j <= w_xJ and 0 <= w_kJ - j <= nm - w_xJ.
    j_lo = max(0, w_kJ - (nm - w_xJ))
    j_hi = min(w_xJ, w_kJ)

    denom = binomial(nm, w_kJ)
    half = mpf(1) / 2

    # Accumulate, per residue m in Z_6, the sum of kept terms.
    sums = [mpf(0)] * 6
    for j in range(j_lo, j_hi + 1):
        term = binomial(w_xJ, j) * binomial(nm - w_xJ, w_kJ - j)
        # term contributes to residue m iff H(m + j) = 1, i.e.
        # (m + j) mod 6 in {3, 4, 5}.
        r = j % 6
        for m in range(6):
            if (m + r) % 6 in _H_ONE_RESIDUES:
                sums[m] += term

    best = mpf(0)
    for m in range(6):
        eta_m = sums[m] / denom - half
        best = max(best, eta_m ** 2)
    return best


# ---------------------------------------------------------------------------
# 3. Section 3.2 -- the aggregated distinguishing attack
# ---------------------------------------------------------------------------
#
# The attack keeps *every* sample, weighting each Hamming-weight class by its
# (squared) bias.  Its data complexity is
#
#   N >= C_thr / d ,     with     d = sum_{w=0}^{n} p_w * epsilon(w, w_k, n)^2.
#
# The paper's reported/experimental numbers use the constant C_thr = 6
# (it writes "N ~ ceil(6/d)"), which we take as the default.  The theoretical
# Hoeffding bound in the paper uses (2.58)^2 ~ 6.66 instead; it is exposed as a
# parameter so either convention can be reproduced.
#
# Time = O(N), Memory = O(1), Query = N.

DEFAULT_AGG_CONSTANT = mpf(6)  # the "6" in N ~ ceil(6/d)
DEFAULT_AGGSPLITMULT_CONSTANT = mpf(2)


def aggregated_d_value(w_k, n):
    """
    The aggregate statistic d = sum_w p_w * epsilon(w, w_k, n)^2 of Section 3.2.
    This is the inverse data complexity (up to the threshold constant).
    """
    d = mpf(0)
    for w in range(0, n + 1):
        p = weight_probability(w, n)
        e = epsilon_bias(w, w_k, n)
        d += p * e * e
    return d


def aggregated_attack(n, w_k, threshold_constant=DEFAULT_AGG_CONSTANT):
    """
    Section 3.2 aggregated attack.

    Parameters
    ----------
    n   : length of the key/input.
    w_k : Hamming weight of the key (the paper's beta*n).

    Returns
    -------
    dict with:
        'N'           : data complexity (mpf, the raw number of queries),
        'log2_data'   : log2 of the data complexity,
        'log2_time'   : log2 of the time complexity (Time = O(N)),
        'log2_memory' : log2 of the memory complexity (Memory = O(1)),
        'd'           : the underlying aggregate statistic d (for inspection).
    """
    d = aggregated_d_value(w_k, n)
    N = threshold_constant / d                 # N >= C / d
    return {
        'N':           N,
        'log2_data':   log2(N),
        'log2_time':   log2(N),                # Time = O(N)
        'log2_memory': 0.0,                    # Memory = O(1)  ->  log2 = 0
        'd':           d,
    }


# ---------------------------------------------------------------------------
# 4. Section 4.3 -- the aggregated splitting strategy
# ---------------------------------------------------------------------------
#
# Here the key is split into I = {0,...,s-1} and J = {s,...,n-1}, with
# nm = n - s the length of J.  We condition on the contribution of kI and
# aggregate the relevant Hamming-weight classes w of xJ that lie within
# sqrt(nm) of a chosen central weight w_xJ (the paper's alpha*(n-s)).  Both s
# and w_xJ are FREE parameters of the attack -- w_xJ is not fixed to (n-s)/2.
# (Section 4.2 notes that w_xJ ~ (n-s)/2 is optimal in the symmetric beta = 1/2
# case, but the code does not assume it.)
#
# The aggregated distinguisher combines all these classes into a single
# statistic, exactly as Algorithm 2 does in Section 3.2.  Its data complexity is
# therefore the *aggregated* (harmonic-style) quantity
#
#                   C
#   N  =  --------------------------- ,
#         sum_w p_w^{(nm)} max_m eta(w)^2
#
# with p_w^{(nm)} = C(nm, w) / 2^{nm}.  This is the direct analogue of the
# Section 3.2 statistic N = C / sum_w p_w epsilon(w)^2, with epsilon replaced by
# the amplified bias max_m eta (we always have max_m eta >= epsilon).
#
# IMPORTANT: the split size s does NOT appear in the data complexity.  Larger s
# only amplifies the bias, which *lowers* N; the price of s is paid in time and
# memory below.  Consequently, as s -> 0 the data complexity smoothly recovers
# the Section 3.2 aggregated attack -- the natural sanity check.
#
# Time   = sqrt(nm) * s * 2^s  +  N
# Memory = s * 2^s
#
# (matching "Data = N, Time = O(sqrt(n-s) s 2^s + N), Memory = O(s 2^s)").

def splitting_relevant_weights(nm, w_xJ):
    """
    The Hamming-weight classes w of xJ kept by the aggregated splitting attack:
    those with |w - w_xJ| < sqrt(nm), clamped to the valid range [0, nm].
    """
    radius = sqrt(mpf(nm))
    lo = max(0, int(w_xJ - radius) - 1)
    hi = min(nm, int(w_xJ + radius) + 1)
    return [w for w in range(lo, hi + 1)
            if abs(w - w_xJ) < radius]


def splitting_d_value(n, s, w_kJ, w_xJ):
    """
    The aggregate statistic d_split for the splitting strategy:

        d_split = sum_{w in window} p_w^{(n-s)} * max_m eta(w, w_kJ, n-s)^2,

    where p_w^{(n-s)} = C(n-s, w) / 2^{n-s} and the window is the set of input
    weights w of xJ with |w - w_xJ| < sqrt(n-s) (centred on w_xJ).

    This is the exact analogue of the Section 3.2 statistic
    d = sum_w p_w * epsilon(w, w_k, n)^2, with two changes coming from the
    splitting strategy:
      * the length is n - s instead of n (part I is conditioned away), and
      * the per-class bias epsilon is replaced by the *amplified* bias
        max_m eta(.)  (we have max_m eta >= epsilon, which is exactly why
        splitting helps).

    Parameters
    ----------
    n    : total length.
    s    : split size.
    w_kJ : Hamming weight of the key on part J (length n - s).
    w_xJ : centre of the aggregation window, i.e. the Hamming weight of xJ
           around which the relevant classes are taken (the paper's
           alpha*(n - s)).  This is a free parameter, NOT fixed to (n-s)/2.

    Returns an mpf.
    """
    nm = n - s
    d = mpf(0)
    for w in splitting_relevant_weights(nm, w_xJ):
        p = weight_probability(w, nm)               # p_w^{(n-s)}
        d += p * eta_max_squared(w, w_kJ, nm)        # + p_w * max_m eta(w)^2
    return d


def splitting_data_complexity(n, s, w_kJ, w_xJ,
                              threshold_constant=(DEFAULT_AGG_CONSTANT*DEFAULT_AGGSPLITMULT_CONSTANT)):
    """
    Data complexity N of the Section 4.3 aggregated splitting strategy.

    By analogy with Section 3.2, the aggregated distinguisher combines all the
    relevant Hamming-weight classes into a single statistic, so the data
    complexity is the *harmonic-style* aggregate

        N = C / d_split,        d_split = sum_w p_w^{(n-s)} max_m eta(w)^2,

    and NOT the term-by-term sum of the individual single-class complexities.

    Crucially, the split size s does NOT enter the data complexity: increasing s
    only amplifies the bias (raising d_split, hence lowering N).  The cost of s
    (the 2^s guessing) shows up in the *time* and *memory*, handled in
    splitting_attack().  As a sanity check, with s -> 0 this collapses onto the
    Section 3.2 aggregated attack.

    Parameters
    ----------
    n    : total length.
    s    : split size (length of part I that we guess/condition on).
    w_kJ : Hamming weight of the key on part J (length n - s).
    w_xJ : centre of the aggregation window over the weight of xJ (free
           parameter; the paper's alpha*(n - s)).

    Returns
    -------
    N : mpf, the raw data complexity (no big-O, no constant C).
    """
    d = splitting_d_value(n, s, w_kJ, w_xJ)
    return threshold_constant / d


def splitting_attack(n, s, w_kJ, w_xJ):
    """
    Section 4.3 aggregated splitting strategy.

    Parameters
    ----------
    n    : total length.
    s    : split size (length of part I that is guessed/conditioned on).
    w_kJ : Hamming weight of the key on part J (length n - s), as assumed by
           Algorithm 3.  This is the weight on J only -- NOT the weight of the
           full-length key.
    w_xJ : centre of the aggregation window over the Hamming weight of xJ, i.e.
           the integer "alpha*(n - s)" of the paper.  Free parameter, supplied
           alongside s.

    Returns
    -------
    dict with log2 of data, time and memory complexities (and the raw N).
    """
    N = splitting_data_complexity(n, s, w_kJ, w_xJ)

    # Cost of the splitting search: guessing/aggregating over 2^s candidates.
    # For s = 0 there is no search, so this cost is simply 0 / negligible and
    # the attack reduces exactly to the Section 3.2 aggregated attack.
    nm = n - s
    if s == 0:
        time_total = N            # Time = O(N)
        memory = mpf(1)           # Memory = O(1)
    else:
        # Time = sqrt(n - s) * s * 2^s + N   (drop big-O / constants).
        time_total = sqrt(mpf(nm)) * mpf(s) * pow2(s) + N
        # Memory = s * 2^s.
        memory = mpf(s) * pow2(s)

    return {
        'N':           N,
        'log2_data':   log2(N),
        'log2_time':   log2(time_total),
        'log2_memory': log2(memory),
    }


def splitting_attack_s(n,alpha,beta, Nlimit):
    """
    Optimise the split size s for the Section 4.3 attack.

    The attack is parametrised by ratios rather than by raw weights: for each
    candidate split size s (with J-length nm = n - s) the integer weights are
    derived as
        w_kJ = round(beta  * nm)      (key weight on part J)
        w_xJ = round(alpha * nm)      (centre of the xJ-weight window)
    and splitting_attack(n, s, w_kJ, w_xJ) is evaluated.  This keeps the
    intended ratios fixed as s varies.

    Parameters
    ----------
    n      : total length.
    alpha  : input-weight ratio (centre of the xJ window is alpha * (n - s)).
    beta   : key-weight ratio on part J (w_kJ = beta * (n - s)).
    Nlimit : selection mode.
               * Nlimit == -1 : minimise the time complexity over s, with no
                                constraint on the data complexity.
               * Nlimit  >  0 : among the s whose data complexity does not
                                exceed Nlimit (a raw query count, NOT a log2),
                                pick the one minimising the time complexity.

    Returns
    -------
    dict : the result of splitting_attack(...) for the chosen s, augmented with
           the keys 's', 'w_kJ', 'w_xJ'.  Returns None if Nlimit > 0 and no
           value of s meets the data constraint.
    """
    best = None
    for s in range(1, n - 1):
        nm = n - s
        w_kJ = int(round(beta * nm))
        w_xJ = int(round(alpha * nm))

        # Skip splits where the derived J-weights are degenerate.
        if not (0 < w_kJ < nm) or not (0 <= w_xJ <= nm):
            continue

        # Early stop: the time complexity is sqrt(nm)*s*2^s + N.  The first
        # term grows monotonically with s while N only shrinks, so once this
        # term alone already exceeds the best total time found, no larger s can
        # improve.  (We only trust this once we have a candidate to compare to.)
        if best is not None:
            time_floor = sqrt(mpf(nm)) * mpf(s) * pow2(s)
            if log2(time_floor) > best['log2_time']:
                break

        res = dict(splitting_attack(n, s, w_kJ, w_xJ))
        res['s'] = s
        res['w_kJ'] = w_kJ
        res['w_xJ'] = w_xJ

        # In constrained mode, reject candidates whose data exceeds Nlimit.
        if Nlimit != -1 and res['N'] > mpf(Nlimit):
            continue

        # Keep the candidate with the smallest time complexity.
        if best is None or res['log2_time'] < best['log2_time']:
            best = res

    return best


# ---------------------------------------------------------------------------
# 5. Main -- report the complexities for a chosen (n, s, w_k)
# ---------------------------------------------------------------------------

def main():
    # ---- Parameters to play with -----------------------------------------
    n    = 514         # key / input length
    #s    = 0          # split size for Section 4.3 (~0.09 n is near-optimal)
    alpha = 0.5
    beta = 0.5
    
    # Section 3.2: key Hamming weight over the full length n.
    w_k  = round(alpha*n)      # typical random key: beta = 1/2
    # Section 4.3: weight of the key on part J (length n - s), and the centre
    # w_xJ of the window over the weight of xJ.  Both are free parameters.
    #w_kJ = round(alpha*(n - s))    # weight of k on J  (beta*(n-s) with beta = 1/2)
    #w_xJ = round(beta*(n - s))    # centre of the xJ-weight window (alpha*(n-s), alpha=1/2)
    # ----------------------------------------------------------------------
    print(f"Parameters: n = {n}")
    print(f"  Section 3.2: beta  = {beta}")
    print(f"  Section 4.3: beta = {beta}, alpha = {alpha}")
    print(f"(all complexities are reported as log2, i.e. as exponents)\n")

    # --- Section 3.2 : aggregated attack (input: n, w_k) ---
    agg = aggregated_attack(n, w_k)
    print("Section 3.2 -- aggregated attack")
    print(f"  log2(data)   = {agg['log2_data']:.2f}")
    print(f"  log2(time)   = {agg['log2_time']:.2f}")
    print(f"  log2(memory) = {agg['log2_memory']:.2f}")

    print()

    ## --- Section 4.3 : aggregated splitting strategy (input: n, s, w_kJ, w_xJ) ---
    #spl = splitting_attack(n, s, w_kJ, w_xJ)
    #print("Section 4.3 -- aggregated splitting strategy")
    #print(f"  log2(data)   = {spl['log2_data']:.2f}")
    #print(f"  log2(time)   = {spl['log2_time']:.2f}")
    #print(f"  log2(memory) = {spl['log2_memory']:.2f}")
    #print()

    # --- Section 4.3 : optimal split size s -------------------------------
    # Mode Nlimit = -1 : minimise the time complexity over s.
    opt = splitting_attack_s(n, alpha,beta, Nlimit=-1)
    print("Section 4.3 -- optimal s (minimise time, regardless of data)")
    print(f"  best s       = {opt['s']}")
    print(f"  log2(data)   = {opt['log2_data']:.2f}")
    print(f"  log2(time)   = {opt['log2_time']:.2f}")
    print(f"  log2(memory) = {opt['log2_memory']:.2f}")

    print()

    # Mode Nlimit = N : minimise time subject to data <= N (here N = 2^45).
    Nlimit = mpf(2) ** (45)
    opt2 = splitting_attack_s(n, alpha,beta, Nlimit=Nlimit)
    print("Section 4.3 -- optimal s (minimise time s.t. data <= 2^(45))")
    if opt2 is None:
        print("  no split size meets the data constraint")
    else:
        print(f"  best s       = {opt2['s']}")
        print(f"  log2(data)   = {opt2['log2_data']:.2f}")
        print(f"  log2(time)   = {opt2['log2_time']:.2f}")
        print(f"  log2(memory) = {opt2['log2_memory']:.2f}")


if __name__ == "__main__":
    main()