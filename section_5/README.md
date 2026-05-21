To use the different programs:

**Comparison of the asymptotic formula for the denominator**

```bash
python3 asymptotic_estimation_denominator.py <n> <w> 
```

**Comparison of the asymptotic formula for the numerator**

```bash
python3 asymptotic_estimation_numerator.py <n> <w>
```

**Display the exact bias over all possible w for a fixed n and beta**

```bash
python3 johansson_var_alpha.py <n> <beta n>
```

**Display the exact bias over all possible beta for a fixed n and w**

```bash
python3 johansson_var_beta.py <n> <w>
```

**Compute the exact bias for n, w, beta and run the corresponding Monte-Carlo simulation**

```bash
g++ -O3 -fopenmp formula_johansson_fixed_weight.cpp -o formula_johansson_fixed_weight
./johansson_fixed_weight <n> <w> <beta n> <nb_samples>
```

**Display the distribution of the different laws that were used in the proofs of asymptotic formulas**

```bash
python3 justify_gaussian_behavior_prop_numerator.py
python3 justify_gaussian_behavior_prop_denominator.py
```

---

## Server Scripts

*The following codes were run on a server using more than 32 cores.*

**Search for the asymptotic complexity of the attack** *(Note: The C++ code must be run before the Python script)*

```bash
g++ -O3 -fopenmp worst_bias_asymp.cpp -o worst_bias_asymp
./worst_bias_asymp
python3 find_attack_complexity_asymp.py
```

