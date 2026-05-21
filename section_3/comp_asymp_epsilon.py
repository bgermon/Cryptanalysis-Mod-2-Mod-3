import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_file = "bias_results.csv"
    
    # Read the data saved by the C++ program
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: the file {csv_file} does not exist yet.")
        return

    # Group the data by each (alpha, beta) pair
    groups = df.groupby(['alpha', 'beta'])

    for (alpha, beta), data in groups:
        data = data.sort_values(by='n')
        
        plt.figure(figsize=(9, 6))
        
        plt.plot(data['n'], data['log2_abs_exact'], 
                 marker='o', linestyle='-', label='Exact Bias')
        
        plt.plot(data['n'], data['log2_abs_approx'], 
                 marker='x', linestyle='--', color='red', label='Asymptotic Bias')
        
        # Format the plot
        plt.title(f"Evolution of the bias logarithm\n$\\alpha={alpha}$ | $\\beta={beta}$", fontsize=14)
        plt.xlabel("n (Sample size)", fontsize=12)
        plt.ylabel("$\\log_2(|bias|)$", fontsize=12)
        
        plt.legend(fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Adjust margins
        plt.tight_layout()
        
        # Display the figure. Closing the window moves to the next figure.
        plt.show()

if __name__ == '__main__':
    main()