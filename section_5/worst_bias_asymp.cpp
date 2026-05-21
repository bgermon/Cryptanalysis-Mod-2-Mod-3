#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <atomic>
#include <algorithm> // Added for std::sort
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <omp.h>

using namespace boost::multiprecision;
typedef cpp_dec_float_100 dec_float;

std::vector<std::vector<dec_float>> C;

void precompute_combinations(int max_n) {
    C.assign(max_n + 1, std::vector<dec_float>(max_n + 1, dec_float("0")));
    for (int i = 0; i <= max_n; ++i) {
        C[i][0] = dec_float("1");
        for (int j = 1; j <= i; ++j) {
            C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
        }
    }
}

dec_float exact_cos_pi_over_6(int m) {
    m = ((m % 12) + 12) % 12;
    dec_float half("0.5");
    dec_float sqrt3_over_2 = sqrt(dec_float("3")) / dec_float("2");
    
    if (m == 0) return dec_float("1");
    else if (m == 1 || m == 11) return sqrt3_over_2;
    else if (m == 2 || m == 10) return half;
    else if (m == 3 || m == 9) return dec_float("0");
    else if (m == 4 || m == 8) return -half;
    else if (m == 5 || m == 7) return -sqrt3_over_2;
    else if (m == 6) return dec_float("-1");
    return dec_float("0");
}

const dec_float PI_SQ("9.86960440108935861883449099987615113531369940724079");

dec_float asymptotic_numerator(int n, dec_float alpha, dec_float beta, int k0) {
    int b_int = std::round(static_cast<double>(beta) * n);
    
    dec_float B1_amp = (dec_float("2") / dec_float("3")) * exact_cos_pi_over_6(b_int - 2) * exact_cos_pi_over_6(k0);
    dec_float B1_pow = -(PI_SQ * n * beta / dec_float("72")) * (dec_float("1") - alpha * beta);
    dec_float B1 = B1_amp * exp(B1_pow);
    
    dec_float B2_amp = (dec_float("2") / dec_float("9")) * exact_cos_pi_over_6(2 * b_int - 4) * exact_cos_pi_over_6(2 * k0);
    dec_float B2_pow = -(PI_SQ * n * beta / dec_float("18")) * (dec_float("1") - alpha);
    dec_float B2 = B2_amp * exp(B2_pow);
    
    dec_float B3_amp = dec_float("2") / dec_float("9");
    dec_float B3_pow = -(PI_SQ * n * alpha * beta / dec_float("18")) * (dec_float("1") - beta);
    dec_float B3 = B3_amp * exp(B3_pow);
    
    return dec_float("0.25") + B1 + B2 + B3;
}

dec_float asymptotic_denominator(int n, dec_float alpha, dec_float beta, int k0) {
    int b = std::round(static_cast<double>(beta) * n);
    dec_float amplitude = (dec_float("2") / dec_float("3")) * exact_cos_pi_over_6(k0) * exact_cos_pi_over_6(-2 + b);
    dec_float variance_term = beta * (dec_float("1") - alpha * beta);
    dec_float power =  -PI_SQ / dec_float(72.0) * variance_term * n;
    
    return amplitude * exp(power) + dec_float("0.5");
}

dec_float worst_case_asymp_bias(int n, int a, int k0, int b_min, int b_max, bool skip_b_mod_6_eq_5) {
    dec_float min_abs_bias("1.0");
    dec_float worst_bias_value("0");
    dec_float alpha = dec_float(a) / dec_float(n);
    bool valid_bias_found = false;
    
    for (int b = b_min; b <= b_max; ++b) {
        // Apply the constraint conditionally based on the flag
        if (skip_b_mod_6_eq_5 && (b % 6 == 5)) continue;

        dec_float beta = dec_float(b) / dec_float(n);
        dec_float num = asymptotic_numerator(n, alpha, beta, k0);
        dec_float den = asymptotic_denominator(n, alpha, beta, k0);
        
        if (den > dec_float("0")) {
            dec_float bias_b = (num / den) - dec_float("0.5");
            dec_float current_abs = abs(bias_b);
            
            if (current_abs < min_abs_bias) {
                min_abs_bias = current_abs;
                worst_bias_value = bias_b;
                valid_bias_found = true;
            }
        }
    }
    return valid_bias_found ? worst_bias_value : dec_float("0");
}

// Updated struct to hold both sets of results
struct Record {
    int n;
    int a;
    dec_float asymp_bias_all;
    dec_float asymp_bias_skip;
};

int main() {
    int n_max = 4096;
    std::cout << "Precomputing Pascal's Triangle for n=" << n_max << "...\n";
    precompute_combinations(n_max);
    
    std::vector<int> n_values;
    for (int n = 256; n <= n_max; n++) {
        n_values.push_back(n);
    }
    
    std::vector<Record> all_results;
    std::atomic<int> completed_n{0};

    std::cout << "Computing Asymptotic Biases for n from 256 to 2048...\n";

    #pragma omp parallel for schedule(dynamic, 1) num_threads(64)
    for (size_t i = 0; i < n_values.size(); ++i) {
        int n = n_values[i];
        
        dec_float cdf("0");
        dec_float two_to_n = pow(dec_float("2"), n);
        int b_min = -1, b_max = -1;
        
        for (int b = 0; b <= n; ++b) {
            cdf += C[n][b] / two_to_n;
            if (b_min == -1 && cdf >= dec_float("0.005")) b_min = b;
            if (b_max == -1 && cdf >= dec_float("0.995")) { b_max = b; break; }
        }
        if (b_max == -1) b_max = n;

        std::vector<Record> local_results;
        for (int a = 0; a <= n; a += 2) {
            // Calculate with both flags
            dec_float asymp_all = worst_case_asymp_bias(n, a, 0, b_min, b_max, false);
            dec_float asymp_skip = worst_case_asymp_bias(n, a, 0, b_min, b_max, true);
            
            local_results.push_back({n, a, asymp_all, asymp_skip});
        }
        
        #pragma omp critical
        {
            all_results.insert(all_results.end(), local_results.begin(), local_results.end());
            completed_n++;
            std::cout << "\rCompleted " << completed_n << " / " << n_values.size() << " 'n' sizes." << std::flush;
        }
    }
    
    std::cout << "\nComputations finished! Sorting results...\n";
    
    // Ensure the output is sequentially ordered by n, then a
    std::sort(all_results.begin(), all_results.end(), [](const Record& r1, const Record& r2) {
        if (r1.n != r2.n) return r1.n < r2.n;
        return r1.a < r2.a;
    });

    std::cout << "Saving to asymptotic_complexity.csv...\n";
    std::ofstream out("asymptotic_complexity.csv");
    out << "n,a,asymp_bias_all,asymp_bias_skip_5_mod_6\n";
    out << std::setprecision(80) << std::scientific;
    for (const auto& r : all_results) {
        out << r.n << "," << r.a << "," << r.asymp_bias_all << "," << r.asymp_bias_skip << "\n";
    }
    out.close();
    
    return 0;
}