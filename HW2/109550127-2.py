import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def binomial_likelihood(m, N, p):
    return gamma(N + 1) / (gamma(m + 1) * gamma(N - m + 1)) * (p ** m) * ((1 - p) ** (N - m))

def beta_distribution(p, a, b):
    return gamma(a + b) / (gamma(a) * gamma(b)) * (p ** (a - 1)) * ((1 - p) ** (b - 1))

def update_posterior(data, a, b):
    m = sum(data)
    N = len(data)
    return a + m, b + N - m

def plot_distributions(a_prior, b_prior, a_posterior, b_posterior, m, N):
    p = np.linspace(0, 1, 100)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(p, beta_distribution(p, a_prior, b_prior), 'r')
    plt.title("Prior")

    plt.subplot(1, 3, 2)
    plt.plot(p, binomial_likelihood(m, N, p), 'b')
    plt.title("Likelihood")
    
    plt.subplot(1, 3, 3)
    plt.plot(p, beta_distribution(p, a_posterior, b_posterior), 'g')
    plt.title("Posterior")
    
    plt.show()

def main():
    a, b = 0, 0
    file_path = "./data.txt"

    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            data = np.array([int(x) for x in line.strip()])
            m = data.sum()
            N = len(data)
            likelihood = binomial_likelihood(m, N, m / N)
            
            print(f"Case {i}:")
            print(f"Data: {line.strip()}")
            print(f"Likelihood: {likelihood:.8f}")
            print(f"Beta prior: a = {a} b = {b}")
            
            # 更新後驗機率
            a_posterior, b_posterior = update_posterior(data, a, b)
            print(f"Beta posterior: a = {a_posterior} b = {b_posterior}\n")
            
           
            plot_distributions(a, b, a_posterior, b_posterior, m, N)
            
            # 更新先驗機率
            a, b = a_posterior, b_posterior

if __name__ == "__main__":
    main()
