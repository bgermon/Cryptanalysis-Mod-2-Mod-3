To use the different programs:

**Compute the exact value of the bias epsilon**
```bash
python3 exact_epsilon.py 
```

**Compare the asymptotic and exact formula**

```bash
g++ -O3 -fopenmp comp_asymp_epsilon.cpp -o comp_asymp_epsilon
./comp_asymp_epsilon
python3 comp_asymp_epsilon.py
```

**Display the exact bias over all possible w for a fixed n and beta**

```bash
python3 johansson_var_alpha.py <n> <beta n>
```

## Server scripts

# The targeted value of n has to be modified in the cpp file
```bash
g++ -O3 -fopenmp aggregated_attack.cpp -o attack
./attack 
```