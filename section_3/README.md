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

**Compute the attack complexities of the aggregated attacks with and without the splitting strategy**

```bash
python3 find_attack_complexity_with_and_without_splitting.py
```


## Server scripts

# The targeted value of n has to be modified in the cpp file
```bash
g++ -O3 -fopenmp aggregated_attack.cpp -o attack
./attack 
```