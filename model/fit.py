import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

# Import everything from your files
from SEIHDR import run_model, ANCESTRAL_PARAMS, OMICRON_PARAMS, DEFAULT_POPULATION

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from data.plot import load_hkgov_critical_series


def objective_function(params_to_fit, t, observed_critical, y0, static_params):
    """
    Calculates the sum of squared errors between the model's total 
    critical cases and the observed data.
    """
    beta = params_to_fit[0]
    
    # Run the ODE solver
    # y_hat shape: (len(t), 18) -> 3 age groups * 6 compartments
    y_hat = run_model(y0, t, beta, static_params)
    
    # Reshape to easily extract the H (hospitalized/critical) compartment
    # Compartment indices: S=0, E=1, I=2, H=3, R=4, D=5
    state_reshaped = y_hat.reshape(len(t), 3, 6)
    model_hospitalized_per_group = state_reshaped[:, :, 3] 
    
    # Sum across all age groups to compare against total HKGov critical data
    model_total_critical = np.sum(model_hospitalized_per_group, axis=1)
    
    # Compute Residual Sum of Squares (RSS)
    rss = np.sum((observed_critical - model_total_critical) ** 2)
    return rss

def main(argv=None):
    parser = argparse.ArgumentParser(description='Ooh fun! Parameter fitting.')
    parser.add_argument('--how', choices=['least_squares'], default='least_squares', help='Which fitting method to use')
    parser.add_argument('--wave', choices=['4', '5'], default='4', help='Wave to fit: 4 (Ancestral) or 5 (Omicron)')
    args = parser.parse_args(argv)

    # 1. Set dates and parameter profiles based on selected wave
    if args.wave == '4':
        start_date = '2020-11-15'
        end_date = '2021-05-15'
        model_params = ANCESTRAL_PARAMS
        wave_title = "Wave 4 (Ancestral)"
    else:
        start_date = '2022-01-01'
        end_date = '2022-06-01'
        model_params = OMICRON_PARAMS
        wave_title = "Wave 5 (Omicron)"

    # 2. Load the real observed target data
    print(f"Loading HK Gov data for {wave_title}...")
    dates, observed_critical = load_hkgov_critical_series(start_date=start_date, end_date=end_date)
    
    # Create an evenly spaced time array mapping to data points (days)
    t = np.arange(len(dates))
    
    if len(observed_critical) == 0:
        raise ValueError("No data returned for the specified date range. Check your CSV path or dates.")

    # 3. Setup Initial Conditions (y0)
    # 3 age groups, 6 compartments each: [S, E, I, H, R, D]
    # Let's seed a small starting number of exposed/infected individuals
    init_exposed = 10.0
    init_infected = 5.0
    init_hospitalized = observed_critical[0] / 3.0 # Distribute baseline across age tiers evenly
    
    y0_matrix = np.zeros((3, 6))
    for i in range(3):
        y0_matrix[i, 0] = DEFAULT_POPULATION[i] - init_exposed - init_infected - init_hospitalized
        y0_matrix[i, 1] = init_exposed
        y0_matrix[i, 2] = init_infected
        y0_matrix[i, 3] = init_hospitalized
        
    y0 = y0_matrix.reshape(-1) # Flatten to 1D vector of length 18

    # 4. Perform Least Squares Fitting
    print(f"Optimizing beta using {args.how}...")
    initial_beta_guess = [0.5] 
    
    # Restrict beta to positive values
    bounds = [(1e-3, 5.0)] 

    result = minimize(
        objective_function,
        initial_beta_guess,
        args=(t, observed_critical, y0, model_params),
        method='L-BFGS-B',
        bounds=bounds
    )

    best_beta = result.x[0]
    print("\n--- Fitting Results ---")
    print(f"Success: {result.success}")
    print(f"Optimized Transmission Rate (beta): {best_beta:.4f}")
    print(f"Final Loss (RSS): {result.fun:.2f}")

    # 5. Generate Fitted Model Curves for Plotting
    fitted_solution = run_model(y0, t, best_beta, model_params)
    fitted_hospitalized = np.sum(fitted_solution.reshape(len(t), 3, 6)[:, :, 3], axis=1)

    # 6. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(dates, observed_critical, 'ro', label='Observed Critical Data', alpha=0.6, markersize=4)
    plt.plot(dates, fitted_hospitalized, 'b-', label=f'Fitted SEIHDR Model ($\\beta$={best_beta:.3f})', linewidth=2)
    
    plt.title(f"SEIHDR Model Calibration via Least Squares - {wave_title}")
    plt.xlabel("Date")
    plt.ylabel("Total Critical Hospitalizations")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plots_dir = Path(__file__).parent / 'fitted_plots'
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f"fitting_wave_{args.wave}.png")
    plt.show()

if __name__ == '__main__':
    main()