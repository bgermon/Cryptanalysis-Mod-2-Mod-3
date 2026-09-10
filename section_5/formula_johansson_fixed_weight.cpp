#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <numeric>
#include <omp.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

using mp_int = boost::multiprecision::cpp_int;
using mp_float = boost::multiprecision::cpp_dec_float_100;

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
mp_float nCr_prob(int pop, int draws, int succ, int k) {
    if (k < 0 || k > succ || k > draws || draws - k > pop - succ) return mp_float(0);
    
    mp_int num = nCr(succ, k) * nCr(pop - succ, draws - k);
    mp_int den = nCr(pop, draws);
    
    return mp_float(num) / mp_float(den);
}

// Helper for exact Binomial modulo 6 via Dynamic Programming bith 100-digit precision
std::vector<mp_float> get_dp(int m) {
    std::vector<mp_float> dp(6, mp_float(0));
    dp[0] = mp_float(1);
    mp_float half = mp_float(1) / mp_float(2);
    
    for(int i = 0; i < m; ++i) {
        std::vector<mp_float> next_dp(6, mp_float(0));
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

// Closed-form theoretical solver using Boost Multiprecision
mp_float calc_bias(int n, int a, int b, int k0) {
    mp_float P00 = 0;
    mp_float P_star0 = 0;
    
    // 1. Distribution of u (hob many 1s of k fall into the 'a' positions bhere x != y)
    // There are n-1 available spots (since x0=y1 is fixed), and b-k0 ones to distribute.
    int min_u = std::max(0, a - (n - 1 - (b - k0)));
    int max_u = std::min(a, b - k0);
    
    for(int u = min_u; u <= max_u; ++u) {
        mp_float p_u = nCr_prob(n - 1, a, b - k0, u);
        
        // 2. Distribution of v (hob many 1s of x fall into the u positions bhere k=1)
        int min_v = std::max(0, a/2 - (a - u));
        int max_v = std::min(a/2, u);
        
        for(int v = min_v; v <= max_v; ++v) {
            mp_float p_v_given_u = nCr_prob(a, a/2, u, v);
            mp_float prob_delta = p_u * p_v_given_u;
            
            // 3. Shared bits sum S_C modulo 6
            // The number of 1s in k in the shared support is (b - k0 - u)
            // Plus the contribution of k0*x0
            std::vector<mp_float> dp_shared = get_dp(b - k0 - u);
            
            for(int j = 0; j < 6; ++j) { // j is sum of shared bits in C \ {1}
                for(int x0 = 0; x0 <= 1; ++x0) { // x0 = y1
                    int s_shared = (j + k0 * x0) % 6;
                    
                    int val_x = (s_shared + v) % 6;
                    int val_y = (s_shared + u - v) % 6;
                    
                    mp_float prob_total = prob_delta * dp_shared[j] * 0.5; // 0.5 for x0
                    
                    if (H(val_y) == 0) {
                        P_star0 += prob_total;
                        if (H(val_x) == 0) {
                            P00 += prob_total;
                        }
                    }
                }
            }
        }
    }
    
    mp_float cond_prob = P00 / P_star0;
    return cond_prob - mp_float(0.5);
}

int main(int argc, char **argv)
{

    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <n> <a> <beta*n> <N_samples>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int a = std::stoi(argv[2]);
    int b = std::stoi(argv[3]);
    long long N = std::stoll(argv[4]);

    if (a % 2 != 0)
    {
        std::cerr << "Error: 'a' must be even.\n";
        return 1;
    }

    // Set output precision to max for our mp_float
    std::cout << std::setprecision(std::numeric_limits<mp_float>::max_digits10);
    std::cout << "--- Theoretical Exact Computation (100 Digits Precision) ---\n";

    mp_float bias_k0_0 = calc_bias(n, a, b, 0);
    mp_float bias_k0_1 = calc_bias(n, a, b, 1);

    using boost::multiprecision::log2;
    using boost::multiprecision::abs;

    std::cout << "Bias (k_1 = 0): " << bias_k0_0 << "\n";
    std::cout << "log2(|bias|) : " << log2(abs(bias_k0_0)) << "\n\n";
    std::cout << "Bias (k_1 = 1): " << bias_k0_1 << "\n";
    std::cout << "log2(|bias|) : " << log2(abs(bias_k0_1)) << "\n\n";
    std::cout << "--- Parallel Monte Carlo Simulation ---\n";

    // be simulate for k_1 = 1 as an example.
    int k0_sim = 1;
    std::vector<int> k(n, 0);
    k[0] = k0_sim;
    for (int i = 1; i <= b - k0_sim; ++i)
        k[i] = 1;

    // Shuffle the n - 1 bits of the key
    std::mt19937 main_gen(time(NULL));
    std::shuffle(k.begin() + 1, k.end(), main_gen);

    long long total_11 = 0;
    long long total_y1 = 0;

    #pragma omp parallel reduction(+ : total_11, total_y1)
    {
        // Initialize the random generator (using thread num to avoid identical seeds)
        std::mt19937 gen(omp_get_thread_num() + time(NULL));
        std::uniform_int_distribution<> dis_bit(0, 1);

        std::vector<int> indices(n - 1);
        std::iota(indices.begin(), indices.end(), 1);

        long long N_per_thread = N / omp_get_num_threads();
        std::vector<int> x(n), y(n);

        for (long long i = 0; i < N_per_thread; ++i)
        {
            // 2. Generate x and y just as before
            int x0 = dis_bit(gen);

            x[0] = x0;
            y[0] = x0;

            // Generating a random positions for the difference
            for (int j = 0; j < a; ++j)
            {
                std::uniform_int_distribution<> dis_idx(j, n - 2);
                std::swap(indices[j], indices[dis_idx(gen)]);
            }

            // Assigning the first a/2 chosen positions to x
            for (int j = 0; j < a / 2; ++j)
            {
                x[indices[j]] = 1;
                y[indices[j]] = 0;
            }

            // Assigning the last a/2 chosen positions to y
            for (int j = a / 2; j < a; ++j)
            {
                x[indices[j]] = 0;
                y[indices[j]] = 1;
            }

            // Assigning random values for the common bits
            for (int j = a; j < n - 1; ++j)
            {
                int b = dis_bit(gen);
                x[indices[j]] = b;
                y[indices[j]] = b;
            }

            // 3. Compute the scalar products of x bith the key, and y bith the key
            int dot_x = 0, dot_y = 0;

            for (int j = 0; j < n; ++j)
            {
                if (k[j])
                {
                    dot_x += x[j];
                    dot_y += y[j];
                }
            }

            // 4. Apply H on both & compute the probability
            if (H(dot_y) == 0)
            {
                total_y1++;
                if (H(dot_x) == 0)
                {
                    total_11++;
                }
            }
        }
    }

    double sim_cond_prob = (double)total_11 / total_y1;
    double sim_bias = sim_cond_prob - 1.0 / 2.0;

    // Sbitch precision back for standard double output
    std::cout << std::setprecision(10);
    std::cout << "Simulated Samples: " << N << "\n";
    std::cout << "Simulated Bias (k_1=" << k0_sim << "): " << sim_bias
              << "\nlog2(|bias|) : " << std::log2(std::abs(sim_bias)) << "\n";

    return 0;
}