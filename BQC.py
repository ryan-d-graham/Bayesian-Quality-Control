import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import tkinter as tk
from tkinter import messagebox, ttk

# Define the prior models
def jeffreys_prior(M, N):
    M = np.clip(M, 1, N-1)
    return np.power(M / N, -0.5) * np.power(1 - M / N, -0.5)

def uniform_prior(M, N):
    return np.ones_like(M) / (N - 1)

def calculate_prior(M_values, N, prior_model):
    if prior_model == "Jeffreys":
        return calculate_jeffreys_prior(M_values, N)
    elif prior_model == "Uniform":
        return uniform_prior(M_values, N)
    else:
        raise ValueError("Unknown prior model")

def calculate_jeffreys_prior(M_values, N):
    prior = jeffreys_prior(M_values, N)
    prior /= np.sum(prior)
    return prior

# Define the likelihood models
def hypergeometric_likelihood(M, m, N, n):
    return comb(M, m) * comb(N-M, n-m) / comb(N, n)

def binomial_likelihood(M, m, N, n):
    p = M / N
    return comb(n, m) * p**m * (1 - p)**(n - m)

def calculate_likelihood(M, m, N, n, likelihood_model):
    if likelihood_model == "Hypergeometric":
        return hypergeometric_likelihood(M, m, N, n)
    elif likelihood_model == "Binomial":
        return binomial_likelihood(M, m, N, n)
    else:
        raise ValueError("Unknown likelihood model")

def calculate_posterior(N, n, m, prior_model, likelihood_model):
    M_values = np.arange(1, N)
    prior = calculate_prior(M_values, N, prior_model)
    
    likelihoods = np.array([calculate_likelihood(M, m, N, n, likelihood_model) for M in M_values])
    
    unnormalized_posterior = likelihoods * prior
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    
    return M_values, prior, posterior

def realistic_expected_cost_accept(M_values, posterior, c, O, P, R, F, N):
    return np.sum((c * M_values + O * M_values + P * M_values**2 - R * (N - M_values) + F) * posterior)

def realistic_expected_cost_reject(I, N, c, D, F):
    R_p = N * c
    return I + R_p + D + F

def find_max_acceptable_m_realistic(N, n, c, O, P, R, F, I, D, prior_model, likelihood_model):
    m = 1
    all_posteriors = []
    while m < n:
        M_values, prior, posterior = calculate_posterior(N, n, m, prior_model, likelihood_model)
        all_posteriors.append((m, posterior))
        accept_cost = realistic_expected_cost_accept(M_values, posterior, c, O, P, R, F, N)
        reject_cost = realistic_expected_cost_reject(I, N, c, D, F)
        if accept_cost >= reject_cost:
            break
        m += 1
    return m - 1, M_values, prior, all_posteriors

def plot_selected_posteriors(M_values, selected_posteriors, prior, N, max_acceptable_m):
    plt.figure(figsize=(12, 6))
    plt.bar(M_values, prior, alpha=0.3, color='black', label=f'Prior Distribution (N={N})')
    for m, posterior in selected_posteriors:
        label = f'm={m}'
        if m == max_acceptable_m:
            label += " (Recommended Cut-off)"
        plt.bar(M_values, posterior, alpha=0.3, label=label)
    plt.xlabel('Number of Defectives (M)')
    plt.ylabel('Posterior Probability')
    plt.title('Posterior Distributions of Number of Defectives\nM: Number of Defectives in Lot, m: Number of Defectives in Sample')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def print_costs_around_m_realistic(N, n, c, O, P, R, F, I, D, m, prior_model, likelihood_model):
    cost_details = []
    selected_posteriors = []
    for m_val in [m-1, m, m+1]:
        if m_val < 1 or m_val >= n:
            continue
        M_values, prior, posterior = calculate_posterior(N, n, m_val, prior_model, likelihood_model)
        selected_posteriors.append((m_val, posterior))
        accept_cost = realistic_expected_cost_accept(M_values, posterior, c, O, P, R, F, N)
        reject_cost = realistic_expected_cost_reject(I, N, c, D, F)
        cost_details.append((m_val, accept_cost, reject_cost, "Accept" if accept_cost < reject_cost else "Reject"))
    return cost_details, selected_posteriors

def update_values():
    global max_acceptable_m, M_values, prior, selected_posteriors, cost_details
    try:
        N = int(lot_entry.get())
        n = int(sample_entry.get())
        c = float(item_entry.get())
        O = float(opp_cost_entry.get())
        P = float(penalty_entry.get())
        R = float(revenue_entry.get())
        F = float(fixed_cost_entry.get())
        I = float(inspect_entry.get())
        D = float(delay_entry.get())
        prior_model = prior_model_var.get()
        likelihood_model = likelihood_model_var.get()
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values")
        return
    
    max_acceptable_m, M_values, prior, all_posteriors = find_max_acceptable_m_realistic(
        N, n, c, O, P, R, F, I, D, prior_model, likelihood_model)
    cost_details, selected_posteriors = print_costs_around_m_realistic(
        N, n, c, O, P, R, F, I, D, max_acceptable_m, prior_model, likelihood_model)
    
    cost_texts = []
    for detail in cost_details:
        m_val, accept_cost, reject_cost, decision = detail
        cost_texts.append(f"m={m_val}: Accept Cost: {accept_cost:.2f}, Reject Cost: {reject_cost:.2f}, Decision: {decision}")

    output_text.set("\n".join(cost_texts))

def plot_values():
    plot_selected_posteriors(M_values, selected_posteriors, prior, len(M_values) + 1, max_acceptable_m)

# GUI setup with default values
root = tk.Tk()
root.title("Decision-Theoretic Quality Control")

tk.Label(root, text="Total number of items in the lot").grid(row=0, column=0, padx=10, pady=5)
tk.Label(root, text="Sample size").grid(row=1, column=0, padx=10, pady=5)
tk.Label(root, text="Cost per defective item").grid(row=2, column=0, padx=10, pady=5)
tk.Label(root, text="Opportunity cost per defective item").grid(row=3, column=0, padx=10, pady=5)
tk.Label(root, text="Penalty cost coefficient for defectives").grid(row=4, column=0, padx=10, pady=5)
tk.Label(root, text="Revenue per non-defective item").grid(row=5, column=0, padx=10, pady=5)
tk.Label(root, text="Fixed overhead costs").grid(row=6, column=0, padx=10, pady=5)
tk.Label(root, text="Cost of inspection").grid(row=7, column=0, padx=10, pady=5)
tk.Label(root, text="Cost of delay or disruption due to rejection").grid(row=8, column=0, padx=10, pady=5)

lot_entry = tk.Entry(root)
sample_entry = tk.Entry(root)
item_entry = tk.Entry(root)
opp_cost_entry = tk.Entry(root)
penalty_entry = tk.Entry(root)
revenue_entry = tk.Entry(root)
fixed_cost_entry = tk.Entry(root)
inspect_entry = tk.Entry(root)
delay_entry = tk.Entry(root)

# Setting default values
lot_entry.insert(0, "1000")
sample_entry.insert(0, "20")
item_entry.insert(0, "10")
opp_cost_entry.insert(0, "5")
penalty_entry.insert(0, "1")
revenue_entry.insert(0, "20")
fixed_cost_entry.insert(0, "500")
inspect_entry.insert(0, "100")
delay_entry.insert(0, "300")

lot_entry.grid(row=0, column=1, padx=10, pady=5)
sample_entry.grid(row=1, column=1, padx=10, pady=5)
item_entry.grid(row=2, column=1, padx=10, pady=5)
opp_cost_entry.grid(row=3, column=1, padx=10, pady=5)
penalty_entry.grid(row=4, column=1, padx=10, pady=5)
revenue_entry.grid(row=5, column=1, padx=10, pady=5)
fixed_cost_entry.grid(row=6, column=1, padx=10, pady=5)
inspect_entry.grid(row=7, column=1, padx=10, pady=5)
delay_entry.grid(row=8, column=1, padx=10, pady=5)

tk.Label(root, text="Prior Model").grid(row=9, column=0, padx=10, pady=5)
tk.Label(root, text="Likelihood Model").grid(row=10, column=0, padx=10, pady=5)

prior_model_var = tk.StringVar(value="Jeffreys")
prior_model_menu = ttk.Combobox(root, textvariable=prior_model_var, values=["Jeffreys", "Uniform"])
prior_model_menu.grid(row=9, column=1, padx=10, pady=5)

likelihood_model_var = tk.StringVar(value="Hypergeometric")
likelihood_model_menu = ttk.Combobox(root, textvariable=likelihood_model_var, values=["Hypergeometric", "Binomial"])
likelihood_model_menu.grid(row=10, column=1, padx=10, pady=5)

tk.Button(root, text='Update', command=update_values).grid(row=11, column=0, columnspan=1, padx=10, pady=10)
tk.Button(root, text='Plot', command=plot_values).grid(row=11, column=1, columnspan=1, padx=10, pady=10)

output_text = tk.StringVar()
output_label = tk.Label(root, textvariable=output_text, justify=tk.LEFT)
output_label.grid(row=12, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
