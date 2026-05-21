#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <omp.h>

// BigInt for pure and exact integer combinatorics
using BigInt = boost::multiprecision::cpp_int;

// 300 decimal places of precision
using Real = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<300>>;

BigInt nCr(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k;

    BigInt res = 1;
    for (int i = 1; i <= k; ++i) {
        res *= (n - i + 1);
        res /= i; 
    }
    return res;
}

Real exact_bias(int n, Real alpha, Real beta) {
    int alphan = (alpha * n).convert_to<int>();
    int betan  = (beta * n).convert_to<int>();

    int j_min = std::max(0, alphan - (n - betan));
    int j_max = std::min(alphan, betan);

    BigInt ctr = 0;

    for (int j = j_min; j <= j_max; ++j) {
        int mod6 = j % 6;
        if (mod6 >= 3 && mod6 <= 5) {
            ctr += nCr(betan, j) * nCr(n - betan, alphan - j);
        }
    }

    BigInt total_comb = nCr(n, alphan);

    if (total_comb == 0) return Real(0);

    BigInt exact_numerator = (2 * ctr) - total_comb;
    BigInt exact_denominator = 2 * total_comb;

    Real r_num(exact_numerator);
    Real r_den(exact_denominator);

    return r_num / r_den;
}

Real approx_bias(int n, Real alpha, Real beta) {
    Real pi = acos(Real(-1));

    Real mu = alpha * beta;
    Real sigma2 = alpha * beta * (1 - alpha) * (1 - beta);

    return Real(1) / 6
        * exp(-(pi*pi*sigma2 / 18) * n)
        * cos((pi / 3) * mu * n + 2*pi / 3);
}

std::string sign_str(const Real& x) {
    if (x > 0) return "+";
    if (x < 0) return "-";
    return "0";
}

Real log2_abs(const Real& x) {
    if (x == 0) return -std::numeric_limits<double>::infinity();
    return log(abs(x)) / log(Real(2));
}

int main() {
    std::vector<Real> vals = {
        Real("0.3"),
        Real("0.4"),
        Real("0.5"),
        Real("0.6"),
        Real("0.7"),
    };

    std::ofstream outfile("bias_results.csv");
    outfile << std::setprecision(80);
    outfile << "n,alpha,beta,sign_exact,log2_abs_exact,sign_approx,log2_abs_approx\n";

    std::cout << "Starting parallel calculations..." << std::endl;

    int num_vals = vals.size();
    int n_start = 1000;
    int n_end = 10000;
    int n_step = 100;
    int num_n = (n_end - n_start) / n_step + 1;

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (int i_n = 0; i_n < num_n; ++i_n) {
        for (int i_a = 0; i_a < num_vals; ++i_a) {
            for (int i_b = 0; i_b < num_vals; ++i_b) {
                
                int n = n_start + i_n * n_step;
                Real alpha = vals[i_a];
                Real beta = vals[i_b];

                Real eps_exact = exact_bias(n, alpha, beta);
                Real eps_approx = approx_bias(n, alpha, beta);

                #pragma omp critical
                {
                    outfile << n << ","
                            << alpha << ","
                            << beta << ","
                            << sign_str(eps_exact) << ","
                            << log2_abs(eps_exact) << ","
                            << sign_str(eps_approx) << ","
                            << log2_abs(eps_approx)
                            << "\n";
                    
                    outfile.flush();
                    std::cout << "Done: n=" << n << " | alpha=" << alpha << " | beta=" << beta << std::endl;
                }
            }
        }
    }

    outfile.close();
    std::cout << "All calculations are finished." << std::endl;
    return 0;
}