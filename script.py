import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import argparse

def jeffreys_prior(M, N):
    # Avoid division by zero and invalid operations for M=0 or M=N
    M = np.clip(M, 1e-10, N-1e-10)
    return np.power(M / N, -0.5) * np.power(1 - M / N, -0.5)

def calculate_jeffreys_prior(M_values, N):
    prior = jeffreys_prior(M_values, N)
    prior /= np.sum(prior)  # Normalize the prior
    return prior

def likelihood(M, m, N, n):
    return comb(M, m) * comb(N-M, n-m) / comb(N, n)

def calculate_posterior(N, n, m):
    M_values = np.arange(0, N+1)
    prior = calculate_jeffreys_prior(M_values, N)
    
    likelihoods = np.array([likelihood(M, m, N, n) for M in M_values])
    
    unnormalized_posterior = likelihoods * prior
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    
    return M_values, prior, posterior

def improved_expected_cost_accept(M_values, posterior, c, P, R, N):
    return np.sum((c * M_values + P * M_values**2 - R * (N - M_values)) * posterior)

def improved_expected_cost_reject(N, c, I, D):
    return N * c + I + D

def find_max_acceptable_m_improved(N, n, c, P, R, I, D):
    m = 1  # Start from m=1 to skip m=0
    all_posteriors = []
    while m < n:  # Skip m=n
        M_values, prior, posterior = calculate_posterior(N, n, m)
        all_posteriors.append((m, posterior))
        accept_cost = improved_expected_cost_accept(M_values, posterior, c, P, R, N)
        reject_cost = improved_expected_cost_reject(N, c, I, D)
        if accept_cost >= reject_cost:
            break
        m += 1
    return m - 1, M_values, prior, all_posteriors  # Return the last acceptable m value, M_values, prior, and all posteriors

def plot_prior(M_values, prior, N):
    plt.figure(figsize=(12, 6))
    plt.bar(M_values, prior, alpha=0.7, label=f'Jeffreys Prior (N={N})')
    plt.xlabel('Number of Defectives (M)')
    plt.ylabel('Prior Probability')
    plt.title(f'Jeffreys Prior Distribution (N={N})')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_posteriors(M_values, all_posteriors):
    plt.figure(figsize=(12, 6))
    for m, posterior in all_posteriors:
        plt.bar(M_values, posterior, alpha=0.3, label=f'm={m}')
    plt.xlabel('Number of Defectives (M)')
    plt.ylabel('Posterior Probability')
    plt.title('Posterior Distribution of Number of Defectives')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def print_costs_around_m_improved(N, n, c, P, R, I, D, m):
    for m_val in [m-1, m, m+1]:
        if m_val < 1 or m_val >= n:
            continue
        M_values, prior, posterior = calculate_posterior(N, n, m_val)
        accept_cost = improved_expected_cost_accept(M_values, posterior, c, P, R, N)
        reject_cost = improved_expected_cost_reject(N, c, I, D)
        print(f"For m={m_val}:")
        print(f"  Expected Accept Cost: {accept_cost:.2f}")
        print(f"  Expected Reject Cost: {reject_cost:.2f}")
        if accept_cost < reject_cost:
            print(f"  Decision: Accept the lot.")
        else:
            print(f"  Decision: Reject the lot.")

def main():
    parser = argparse.ArgumentParser(description='Calculate and plot the posterior distribution of the number of defectives and find the maximum m value for acceptance.')
    parser.add_argument('--lot', type=int, default=1000, help='Total number of items in the lot')
    parser.add_argument('--sample', type=int, default=20, help='Sample size')
    parser.add_argument('--item', type=float, default=10, help='Cost per defective item')
    parser.add_argument('--penalty', type=float, default=1, help='Penalty cost coefficient for defectives')
    parser.add_argument('--revenue', type=float, default=20, help='Revenue per non-defective item')
    parser.add_argument('--inspect', type=float, default=100, help='Cost of inspection')
    parser.add_argument('--delay', type=float, default=200, help='Cost of delay or disruption due to rejection')
    parser.add_argument('--plot_prior', type=bool, default=True, help='Whether to plot the prior distribution')
    parser.add_argument('--plot_post', type=bool, default=True, help='Whether to plot the posterior distributions')
    
    args = parser.parse_args()
    
    N = args.lot
    n = args.sample
    c = args.item
    P = args.penalty
    R = args.revenue
    I = args.inspect
    D = args.delay
    plot_prior_flag = args.plot_prior
    plot_post_flag = args.plot_post
    
    max_acceptable_m, M_values, prior, all_posteriors = find_max_acceptable_m_improved(N, n, c, P, R, I, D)
    print(f"Maximum acceptable m value: {max_acceptable_m}")
    
    print_costs_around_m_improved(N, n, c, P, R, I, D, max_acceptable_m)
    
    if plot_prior_flag:
        plot_prior(M_values, prior, N)
    
    if plot_post_flag:
        plot_posteriors(M_values, all_posteriors)

if __name__ == "__main__":
    main()
