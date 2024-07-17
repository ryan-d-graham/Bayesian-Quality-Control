```markdown
# Decision-Theoretic Quality Control

This project provides a decision-theoretic quality control framework using a Bayesian approach with Jeffreys Prior and realistic cost functions. The goal is to determine whether to accept or reject a batch of items based on observed sample defect rates. This application is implemented with a Tkinter GUI to allow for user-friendly interaction and visualization of the decision-making process.

## Features

- Bayesian quality control with Jeffreys Prior
- Calculation of posterior distributions based on observed sample defect rates
- Realistic cost models for acceptance and rejection decisions
- Visualization of posterior distributions for selected \( m \)-values
- User-friendly GUI for input and visualization

## Requirements

- Python 3.x
- numpy
- matplotlib
- scipy
- tkinter (usually comes pre-installed with Python)

## Usage

### Running the Application

To run the application, simply execute the script:

```sh
python script.py
```

### GUI Inputs

- **Total number of items in the lot**: The total number of items in the batch.
- **Sample size**: The number of items sampled from the batch for inspection.
- **Cost per defective item**: The cost associated with each defective item.
- **Opportunity cost per defective item**: The cost of lost opportunities due to defective items.
- **Penalty cost coefficient for defectives**: The penalty cost for defective items, which increases quadratically.
- **Revenue per non-defective item**: The revenue generated from each non-defective item.
- **Fixed overhead costs**: The fixed costs associated with the batch.
- **Cost of inspection**: The cost of inspecting the batch.
- **Cost of reprocessing or scrapping the lot**: The cost of reprocessing or scrapping the batch.
- **Cost of delay or disruption due to rejection**: The cost of delays or disruptions caused by rejecting the batch.

### GUI Buttons

- **Update**: Calculate the decision-theoretic quality control results based on the input values.
- **Plot**: Visualize the posterior distributions for the selected \( m \)-values.

### Example

1. Enter the following example values into the GUI:
    - Total number of items in the lot: `100`
    - Sample size: `10`
    - Cost per defective item: `1`
    - Opportunity cost per defective item: `2`
    - Penalty cost coefficient for defectives: `0.5`
    - Revenue per non-defective item: `5`
    - Fixed overhead costs: `50`
    - Cost of inspection: `10`
    - Cost of reprocessing or scrapping the lot: `20`
    - Cost of delay or disruption due to rejection: `15`
    
2. Click the **Update** button to calculate the quality control results.
3. Click the **Plot** button to visualize the posterior distributions for the selected \( m \)-values.

## Code Overview

### Functions

- `jeffreys_prior(M, N)`: Computes Jeffreys Prior for the given number of defectives \( M \) and lot size \( N \).
- `calculate_jeffreys_prior(M_values, N)`: Normalizes Jeffreys Prior for the given range of defectives.
- `likelihood(M, m, N, n)`: Computes the likelihood of observing \( m \) defectives in a sample of size \( n \).
- `calculate_posterior(N, n, m)`: Computes the posterior distribution for the number of defectives based on the observed sample data.
- `realistic_expected_cost_accept(M_values, posterior, c, O, P, R, F, N)`: Calculates the expected cost of accepting the batch.
- `realistic_expected_cost_reject(I, R_p, D, F)`: Calculates the expected cost of rejecting the batch.
- `find_max_acceptable_m_realistic(N, n, c, O, P, R, F, I, R_p, D)`: Finds the maximum acceptable number of defectives based on the cost models.
- `plot_selected_posteriors(M_values, selected_posteriors)`: Plots the selected posterior distributions.
- `print_costs_around_m_realistic(N, n, c, O, P, R, F, I, R_p, D, m)`: Prints the costs and decisions around the selected \( m \)-values.
- `update_values()`: Updates the calculation based on the GUI input values.
- `plot_values()`: Plots the selected posterior distributions.

### GUI Setup

The GUI is set up using Tkinter, with input fields for each parameter and buttons to update and plot the results. The output is displayed in a text label, showing the calculated costs and decisions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```
