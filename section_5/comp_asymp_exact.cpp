#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

using mp_int = boost::multiprecision::cpp_int;
using dec_float = boost::multiprecision::cpp_dec_float_100;

// High-precision pi^2 constant
const dec_float PI_SQ("9.86960440108935861883449099987615113531369940724079");

// Helper to compute exact combinations (nCr) using arbitrary precision integers
mp_int nCr(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k;
    
    mp_int res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1);
        res = res / i;
    }
    return res;
}

// Hypergeometric Probability P(X = k) returning a 100-digit precision float
dec_float nCr_prob(int pop, int draws, int succ, int k) {
    if (k < 0 || k > succ || k > draws || draws - k > pop - succ) return dec_float(0);
    
    mp_int num = nCr(succ, k) * nCr(pop - succ, draws - k);
    mp_int den = nCr(pop, draws);
    
    return dec_float(num) / dec_float(den);
}

// Helper for exact Binomial modulo 6 via Dynamic Programming with 100-digit precision
std::vector<dec_float> get_dp(int m) {
    std::vector<dec_float> dp(6, dec_float(0));
    dp[0] = dec_float(1);
    dec_float half = dec_float(1) / dec_float(2);
    
    for(int i = 0; i < m; ++i) {
        std::vector<dec_float> next_dp(6, dec_float(0));
        for(int j = 0; j < 6; ++j) {
            next_dp[j] = half * dp[j] + half * dp[(j + 5) % 6]; 
        }
        dp = next_dp;
    }
    return dp;
}

// H function defined over Z_6
int H(int r) {
    r = ((r % 6) + 6) % 6;
    return (r >= 3 && r <= 5) ? 1 : 0;
}

// Exact algebraic value of cos(m * pi / 6)
dec_float exact_cos_pi_over_6(int m) {
    // Ensure m is correctly wrapped around positive 12
    m = ((m % 12) + 12) % 12;
    dec_float half("0.5");
    dec_float sqrt3 = boost::multiprecision::sqrt(dec_float("3"));
    dec_float sqrt3_over_2 = sqrt3 / dec_float("2");
    
    if (m == 0) return dec_float("1");
    if (m == 1 || m == 11) return sqrt3_over_2;
    if (m == 2 || m == 10) return half;
    if (m == 3 || m == 9) return dec_float("0");
    if (m == 4 || m == 8) return -half;
    if (m == 5 || m == 7) return -sqrt3_over_2;
    if (m == 6) return dec_float("-1");
    
    return dec_float("0");
}

// Computes the asymptotic approximation for the numerator
dec_float asymptotic_numerator(dec_float n, dec_float alpha, dec_float beta, int k0, int b_int) {
    // Mode 1
    dec_float cos_1a = exact_cos_pi_over_6(b_int - 2);
    dec_float cos_1b = exact_cos_pi_over_6(k0);
    dec_float B1_amp = (dec_float("2") / dec_float("3")) * cos_1a * cos_1b;
    dec_float B1_pow = -(PI_SQ * n * beta / dec_float("72")) * (dec_float("1") - alpha * beta);
    dec_float B1 = B1_amp * boost::multiprecision::exp(B1_pow);
    
    // Mode 2
    dec_float cos_2a = exact_cos_pi_over_6(2 * b_int - 4);
    dec_float cos_2b = exact_cos_pi_over_6(2 * k0);
    dec_float B2_amp = (dec_float("2") / dec_float("9")) * cos_2a * cos_2b;
    dec_float B2_pow = -(PI_SQ * n * beta / dec_float("18")) * (dec_float("1") - alpha);
    dec_float B2 = B2_amp * boost::multiprecision::exp(B2_pow);
    
    // Mode 3
    dec_float B3_amp = dec_float("2") / dec_float("9");
    dec_float B3_pow = -(PI_SQ * n * alpha * beta / dec_float("18")) * (dec_float("1") - beta);
    dec_float B3 = B3_amp * boost::multiprecision::exp(B3_pow);
    
    // Total approximated probability
    return dec_float("0.25") + B1 + B2 + B3;
}

// Computes the asymptotic approximation for the denominator
dec_float asymptotic_denominator(dec_float n, dec_float alpha, dec_float beta, int k0, int b_int) {
    dec_float amplitude = (dec_float("2") / dec_float("3")) * exact_cos_pi_over_6(k0) * exact_cos_pi_over_6(b_int - 2);
    dec_float variance_term = beta * (dec_float("1") - alpha * beta);
    dec_float power =  -PI_SQ / dec_float(72.0) * variance_term * n;
    
    return amplitude * exp(power) + dec_float("0.5");
}

// Wrapper to compute the final asymptotic bias
dec_float asymptotic_bias_calc(int n, int a, int b, int k0) {
    dec_float n_dec(n);
    dec_float alpha = dec_float(a) / n_dec;
    dec_float beta = dec_float(b) / n_dec;

    dec_float num = asymptotic_numerator(n_dec, alpha, beta, k0, b);
    dec_float den = asymptotic_denominator(n_dec, alpha, beta, k0, b);

    if (den == dec_float("0")) {
        return dec_float("0");
    }

    dec_float prob = num / den;
    return prob - dec_float("0.5");
}

// Closed-form theoretical solver using Boost Multiprecision
dec_float calc_bias(int n, int a, int W, int k0) {
    dec_float P11 = 0;
    dec_float P_star1 = 0;
    
    int min_u = std::max(0, a - (n - 1 - (W - k0)));
    int max_u = std::min(a, W - k0);
    
    for(int u = min_u; u <= max_u; ++u) {
        dec_float p_u = nCr_prob(n - 1, a, W - k0, u);
        
        int min_v = std::max(0, a/2 - (a - u));
        int max_v = std::min(a/2, u);
        
        for(int v = min_v; v <= max_v; ++v) {
            dec_float p_v_given_u = nCr_prob(a, a/2, u, v);
            dec_float prob_delta = p_u * p_v_given_u;
            
            std::vector<dec_float> dp_shared = get_dp(W - k0 - u);
            
            for(int j = 0; j < 6; ++j) {
                for(int x1 = 0; x1 <= 1; ++x1) {
                    int s_shared = (j + k0 * x1) % 6;
                    
                    int val_x = (s_shared + v) % 6;
                    int val_y = (s_shared + u - v) % 6;
                    
                    dec_float prob_total = prob_delta * dp_shared[j] * dec_float("0.5");
                    
                    if (H(val_y) == 0) {
                        P_star1 += prob_total;
                        if (H(val_x) == 0) {
                            P11 += prob_total;
                        }
                    }
                }
            }
        }
    }
    
    dec_float cond_prob = P11 / P_star1;
    return cond_prob - dec_float("0.5");
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <n> <a> <beta*n>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int a = std::stoi(argv[2]);
    int W = std::stoi(argv[3]);

    if (a % 2 != 0)
    {
        std::cerr << "Error: 'a' must be even.\n";
        return 1;
    }

    // Set output precision to max for our dec_float
    std::cout << std::setprecision(std::numeric_limits<dec_float>::max_digits10);
    
    using boost::multiprecision::log2;
    using boost::multiprecision::abs;

    // --- EXACT COMPUTATION ---
    std::cout << "--- Theoretical Exact Computation (100 Digits Precision) ---\n";
    
    dec_float exact_bias_k0_0 = calc_bias(n, a, W, 0);
    std::cout << "Exact Bias (k_1 = 0): " << exact_bias_k0_0 << "\n";
    std::cout << "log2(|bias|)        : " << log2(abs(exact_bias_k0_0)) << "\n\n";
    
    dec_float exact_bias_k0_1 = calc_bias(n, a, W, 1);
    std::cout << "Exact Bias (k_1 = 1): " << exact_bias_k0_1 << "\n";
    std::cout << "log2(|bias|)        : " << log2(abs(exact_bias_k0_1)) << "\n\n";

    // --- ASYMPTOTIC COMPUTATION ---
    std::cout << "--- Asymptotic Approximation (100 Digits Precision) ---\n";

    dec_float approx_bias_k0_0 = asymptotic_bias_calc(n, a, W, 0);
    std::cout << "Approx Bias (k_1 = 0): " << approx_bias_k0_0 << "\n";
    if (approx_bias_k0_0 != 0)
        std::cout << "log2(|bias|)         : " << log2(abs(approx_bias_k0_0)) << "\n\n";
    else
        std::cout << "log2(|bias|)         : -inf\n\n";

    dec_float approx_bias_k0_1 = asymptotic_bias_calc(n, a, W, 1);
    std::cout << "Approx Bias (k_1 = 1): " << approx_bias_k0_1 << "\n";
    if (approx_bias_k0_1 != 0)
        std::cout << "log2(|bias|)         : " << log2(abs(approx_bias_k0_1)) << "\n\n";
    else
        std::cout << "log2(|bias|)         : -inf\n\n";

    return 0;
}