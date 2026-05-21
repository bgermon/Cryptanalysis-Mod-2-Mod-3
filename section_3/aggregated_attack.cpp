#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>
#include <limits>
#include <atomic>
#include <string>
#include <chrono>
#include <thread>
#include <omp.h> // Required for OpenMP

using Real = boost::multiprecision::number<
    boost::multiprecision::cpp_dec_float<200>
>;

// Boolean function H
int H(int w) {
    int r = w % 6;
    return (r == 3 || r == 4 || r == 5) ? 1 : 0;
}

// Compute exact bias for a given weight
Real exact_bias_weight(int n, int h, int w) {
    int L = std::max(0, w - (n - h));
    int U = std::min(w, h);
    if (L > U) return Real(0);

    int mode = ((w + 1) * (h + 1)) / (n + 2);
    mode = std::max(L, std::min(U, mode));

    std::vector<Real> rel(U - L + 1);
    rel[mode - L] = 1;

    for (int i = mode; i < U; i++) {
        Real ratio =
            Real(h - i) / Real(i + 1) *
            Real(w - i) / Real(n - h - w + i + 1);
        rel[i + 1 - L] = rel[i - L] * ratio;
    }

    for (int i = mode; i > L; i--) {
        Real ratio =
            Real(i) / Real(h - i + 1) *
            Real(n - h - w + i) / Real(w - i + 1);
        rel[i - 1 - L] = rel[i - L] * ratio;
    }

    Real denom = 0;
    Real signed_sum = 0;

    for (int i = L; i <= U; i++) {
        Real p = rel[i - L];
        denom += p;
        signed_sum += (H(i) == 1) ? p : -p;
    }

    return signed_sum / (2 * denom);
}

// Generate a random key with weight h
std::vector<uint64_t> random_key(int n, int h, std::mt19937_64 &rng) {
    int words = (n + 63) / 64;
    std::vector<uint64_t> k(words, 0);

    std::vector<int> positions(n);
    std::iota(positions.begin(), positions.end(), 0);
    std::shuffle(positions.begin(), positions.end(), rng);

    for (int i = 0; i < h; i++) {
        int pos = positions[i];
        k[pos / 64] |= (uint64_t(1) << (pos % 64));
    }

    return k;
}

// Generate a random input of length n
std::vector<uint64_t> random_input(int n, std::mt19937_64 &rng) {
    int words = (n + 63) / 64;
    std::vector<uint64_t> x(words);

    for (int i = 0; i < words; i++) {
        x[i] = rng();
    }

    int excess = 64 * words - n;
    if (excess > 0) {
        x.back() &= (~uint64_t(0)) >> excess;
    }

    return x;
}

// Compute the hamming weight of a bit array
int hamming_weight(const std::vector<uint64_t> &x) {
    int w = 0;
    for (uint64_t v : x) {
        w += __builtin_popcountll(v);
    }
    return w;
}

// Compute the intersection weight (bitwise AND)
int intersection_weight(
    const std::vector<uint64_t> &x,
    const std::vector<uint64_t> &k
) {
    int w = 0;
    for (size_t i = 0; i < x.size(); i++) {
        w += __builtin_popcountll(x[i] & k[i]);
    }
    return w;
}

// Evaluation of the pseudo-random function
int F_value(
    const std::vector<uint64_t> &x,
    const std::vector<uint64_t> &k
) {
    return H(intersection_weight(x, k));
}

struct Result {
    long double r;
    long double M;
    long double Z;
    bool classified_as_prf;
};

// Execute a single trial parallelized with OpenMP
Result run_trial(
    int n,
    int h,
    uint64_t N,
    const std::vector<long double> &eps,
    bool prf_case,
    uint64_t seed,
    int current_trial,
    int total_trials,
    const std::string& case_name
) {
    // Base RNG for key generation
    std::mt19937_64 base_rng(seed);
    auto key = random_key(n, h, base_rng);

    long double total_r = 0.0L;
    long double total_M = 0.0L;

    std::atomic<uint64_t> progress(0);
    std::atomic<bool> done(false);

    // Monitor thread for the progress bar of the current trial
    std::thread monitor([&]() {
        const int bar_width = 50;
        while (!done.load(std::memory_order_relaxed)) {
            uint64_t p = progress.load(std::memory_order_relaxed);
            double pct = (double)p / N;
            if (pct > 1.0) pct = 1.0;
            int pos = bar_width * pct;

            // Clear line and print the progress bar
            std::cout << "\33[2K\r" << std::left << std::setw(8) << case_name 
                      << " Trial " << current_trial << "/" << total_trials << " [";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::fixed << std::setprecision(1) << (pct * 100.0) << "%" << std::flush;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // We only update the atomic progress counter every so often to minimize overhead
    uint64_t update_interval = std::max<uint64_t>(1ULL, N / 1000ULL);

    // OpenMP Parallel region
    #pragma omp parallel
    {
        // Give each thread a unique RNG instance seeded with thread ID to avoid race conditions
        int tid = omp_get_thread_num();
        std::mt19937_64 local_rng(seed + tid + 0x9e3779b97f4a7c15ULL);
        std::bernoulli_distribution bit(0.5);

        long double local_r = 0.0L;
        long double local_M = 0.0L;
        uint64_t local_count = 0;

        // Distribute the loop iterations across available threads
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < N; i++) {
            auto x = random_input(n, local_rng);
            int w = hamming_weight(x);

            int y = prf_case ? F_value(x, key) : bit(local_rng);

            long double e = eps[w];
            long double sign = (y == 0) ? 1.0L : -1.0L;

            local_r += sign * e;
            local_M += e * e;

            local_count++;
            if (local_count % update_interval == 0) {
                progress.fetch_add(update_interval, std::memory_order_relaxed);
            }
        }

        // Add remaining count
        uint64_t remainder = local_count % update_interval;
        if (remainder > 0) {
            progress.fetch_add(remainder, std::memory_order_relaxed);
        }

        // Safely accumulate the thread-local results into the global ones
        #pragma omp critical
        {
            total_r += local_r;
            total_M += local_M;
        }
    }

    // Stop monitor and ensure it displays 100%
    done.store(true, std::memory_order_relaxed);
    monitor.join();

    std::cout << "\33[2K\r" << std::left << std::setw(8) << case_name 
              << " Trial " << current_trial << "/" << total_trials << " [";
    for (int i = 0; i < 50; ++i) std::cout << "=";
    std::cout << "] 100.0% (Done)\n";

    bool classified_as_prf = (total_r < -total_M);
    long double Z = total_r / std::sqrt(total_M);

    return {total_r, total_M, Z, classified_as_prf};
}

// Sequence the trials sequentially (parallelism is now *within* each trial)
int run_experiment(
    const std::string& case_name,
    int n,
    int h,
    uint64_t N,
    const std::vector<long double> &eps,
    bool prf_case,
    int trials,
    std::vector<Result> &results
) {
    results.resize(trials);
    int success_count = 0;
    std::random_device rd;

    for (int t = 0; t < trials; t++) {
        uint64_t seed = ((uint64_t)rd() << 32) ^ rd();
        
        Result res = run_trial(n, h, N, eps, prf_case, seed, t + 1, trials, case_name);
        results[t] = res;

        bool correct = prf_case ? res.classified_as_prf : !res.classified_as_prf;
        if (correct) {
            success_count++;
        }
    }

    // Output raw CSV-like results for this specific experimental case
    std::cout << "case,trial,r,M,Z,classification\n";
    for(int t = 0; t < trials; ++t) {
        const auto& res = results[t];
        std::cout << case_name << "," << t << ","
                  << res.r << ","
                  << res.M << ","
                  << res.Z << ","
                  << (res.classified_as_prf ? "weak PRF" : "Random")
                  << "\n";
    }
    std::cout << "\n";

    return success_count;
}

int main(int argc, char **argv) {
    int n = 256;
    int trials = 10;
    int num_threads = omp_get_max_threads();

    if (argc >= 2) n = std::stoi(argv[1]);
    if (argc >= 3) trials = std::stoi(argv[2]);
    if (argc >= 4) num_threads = std::stoi(argv[3]);

    // Bound threads to 64 maximum as requested
    num_threads = std::min(64, num_threads);
    omp_set_num_threads(num_threads);

    const long double beta = 0.5L;
    const long double c = 6.0L;

    int h = int(std::llround(beta * n));

    std::cout << std::setprecision(20);
    std::cout << "n = " << n << "\n";
    std::cout << "beta = 0.5\n";
    std::cout << "h = " << h << "\n";
    std::cout << "trials = " << trials << "\n";
    std::cout << "threads (OpenMP) = " << num_threads << "\n\n";

    std::cout << "Precomputing exact biases...\n";

    std::vector<long double> eps(n + 1);

    for (int w = 0; w <= n; w++) {
        Real e = exact_bias_weight(n, h, w);
        eps[w] = e.convert_to<long double>();
    }

    std::vector<long double> pw(n + 1);
    pw[0] = std::pow(2.0L, -n);

    for (int w = 0; w < n; w++) {
        pw[w + 1] = pw[w] * (long double)(n - w) / (long double)(w + 1);
    }

    long double d = 0.0L;
    for (int w = 0; w <= n; w++) {
        d += pw[w] * eps[w] * eps[w];
    }

    long double N_ld = std::ceil(c / d);

    if (N_ld > (long double)std::numeric_limits<uint64_t>::max()) {
        std::cout << "Computed N is too large to simulate.\n";
        std::cout << "log2(N) = " << std::log2(N_ld) << "\n";
        std::cout << "This parameter set should be reported theoretically only.\n";
        return 0;
    }

    uint64_t N = static_cast<uint64_t>(N_ld);

    std::cout << "log2(d) = " << std::log2(d) << "\n";
    std::cout << "log2(1/d) = " << -std::log2(d) << "\n";
    std::cout << "constant c = " << c << "\n";
    std::cout << "N = ceil(c/d) = " << N << "\n";
    std::cout << "log2(N) = " << std::log2((long double)N) << "\n";
    std::cout << "predicted separation 2*sqrt(N*d) = "
              << 2.0L * std::sqrt((long double)N * d) << "\n\n";

    std::vector<Result> random_results;
    std::vector<Result> prf_results;

    // Run sequentially over trials, parallelize internally
    int success_random = run_experiment("random", n, h, N, eps, false, trials, random_results);
    int success_prf = run_experiment("weakPRF", n, h, N, eps, true, trials, prf_results);

    std::cout << "Success random = " << success_random << "/" << trials << "\n";
    std::cout << "Success weakPRF = " << success_prf << "/" << trials << "\n";

    return 0;
}