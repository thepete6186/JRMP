import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
import shutil
import time

# Import everything from your files
from SEIHDR import run_model, ANCESTRAL_PARAMS, OMICRON_PARAMS, DEFAULT_POPULATION

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from data.plot import load_hkgov_critical_series


def objective_function(params_to_fit, t, observed_critical, static_params, wave):
    """
    Calculates the sum of squared errors between the model's total 
    critical cases and the observed data. Dynamically reconstructs 
    y0 based on the optimizer's current seed percentage guess.
    """
    # Unpack the parameters the optimizer is tweaking
    beta = params_to_fit[0]
    seed_percentage = params_to_fit[1]
    
    # Reconstruct y0 dynamically inside the optimization loop
    y0_matrix = np.zeros((3, 6))
    for i in range(3):
        init_exposed = 0.0
        init_infected = 0.0
        init_hospitalized = 0.0

        if wave == '5':
            # Seed the middle age group (Index 1) dynamically using the optimizer's guess
            if i == 1:
                init_exposed = DEFAULT_POPULATION[i] * seed_percentage
                init_infected = DEFAULT_POPULATION[i] * seed_percentage
            # Keep other age cohorts clean at t=0
            elif i == 2 or i == 0:
                init_exposed = 0.0
                init_infected = 0.0
        else:
            # Seed Wave 4 across groups using the dynamic seed parameter
            if i == 1: # Middle age gets the core seed guess
                init_exposed = DEFAULT_POPULATION[i] * seed_percentage
                init_infected = DEFAULT_POPULATION[i] * seed_percentage
            elif i == 2 or i == 0: # Others get a much smaller fraction of it
                init_exposed = DEFAULT_POPULATION[i] * (seed_percentage * 0.1)
                init_infected = DEFAULT_POPULATION[i] * (seed_percentage * 0.1)
        
        y0_matrix[i, 1] = init_exposed
        y0_matrix[i, 2] = init_infected
        y0_matrix[i, 3] = init_hospitalized
        y0_matrix[i, 4] = 0.0
        y0_matrix[i, 5] = 0.0
        y0_matrix[i, 0] = DEFAULT_POPULATION[i] - np.sum(y0_matrix[i, 1:])
        
    y0 = y0_matrix.reshape(-1)
    
    # Run the ODE solver
    y_hat = run_model(y0, t, beta, static_params)
    state_reshaped = y_hat.reshape(len(t), 3, 6)
    model_total_critical = np.sum(state_reshaped[:, :, 3], axis=1)
    
    return np.sum((observed_critical - model_total_critical) ** 2)


def archive_old_fitted_plots(plots_dir, pattern="fitting_wave_*.png"):
    """
    Sweeps through the directory and safely moves matching older plots into 
    the archive directory, appending a unique execution timestamp.
    """
    old_dir = plots_dir / "old"
    old_dir.mkdir(parents=True, exist_ok=True)

    for old_plot in plots_dir.glob(pattern):
        if old_plot.is_file():
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            archived_name = f"{old_plot.stem}_{timestamp}{old_plot.suffix}"
            shutil.move(str(old_plot), str(old_dir / archived_name))


def build_y0(seed_percentage, wave):
    """Helper function to cleanly reconstruct y0 for plotting after fitting."""
    y0_matrix = np.zeros((3, 6))
    for i in range(3):
        if wave == '5':
            if i == 1:
                init_exposed = DEFAULT_POPULATION[i] * seed_percentage
                init_infected = DEFAULT_POPULATION[i] * seed_percentage
            else:
                init_exposed = 0.0
                init_infected = 0.0
        else:
            if i == 1:
                init_exposed = DEFAULT_POPULATION[i] * seed_percentage
                init_infected = DEFAULT_POPULATION[i] * seed_percentage
            else:
                init_exposed = DEFAULT_POPULATION[i] * (seed_percentage * 0.1)
                init_infected = DEFAULT_POPULATION[i] * (seed_percentage * 0.1)
                
        y0_matrix[i, 1] = init_exposed
        y0_matrix[i, 2] = init_infected
        y0_matrix[i, 3] = 0.0
        y0_matrix[i, 4] = 0.0
        y0_matrix[i, 5] = 0.0
        y0_matrix[i, 0] = DEFAULT_POPULATION[i] - np.sum(y0_matrix[i, 1:])
    return y0_matrix.reshape(-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Parameter fitting with dynamic initial conditions.')
    parser.add_argument('--how', choices=['least_squares'], default='least_squares', help='Which fitting method to use')
    parser.add_argument('--wave', choices=['4', '5'], default='4', help='Wave to fit: 4 (Ancestral) or 5 (Omicron)')
    args = parser.parse_args(argv)

    if args.wave == '4':
        start_date = '2020-11-15'
        end_date = '2021-05-15'   
        model_params = ANCESTRAL_PARAMS
        wave_title = "Wave 4 (Ancestral)"
    else:
        start_date = '2022-01-01'
        end_date = '2022-06-01'   #I have no clue when omicron starts
        model_params = OMICRON_PARAMS
        wave_title = "Wave 5 (Omicron)"

    print(f"Loading HK Gov data for {wave_title}...")
    dates, observed_critical = load_hkgov_critical_series(start_date=start_date, end_date=end_date)
    t = np.arange(len(dates))
    
    if len(observed_critical) == 0:
        raise ValueError("No data returned for the specified date range.")

    # 4. Perform Least Squares Fitting
    print(f"Optimizing beta AND initial seed percentage using {args.how}...")
    
    # Passing dynamic parameters: [Initial Beta, Initial Seed Percentage Guess]
    initial_guesses = [0.5, 0.001] 
    
    # Boundaries: Beta in [0.001, 5.0], Seed fraction in [0.001%, 5%]
    bounds = [(1e-3, 5.0), (1e-5, 0.05)] 

    result = minimize(
        objective_function,
        initial_guesses,
        args=(t, observed_critical, model_params, args.wave),
        method='L-BFGS-B',
        bounds=bounds
    )

    best_beta = result.x[0]
    best_seed_percentage = result.x[1]

    # Calculate final calibrated initial counts for terminal reporting
    final_y0_flat = build_y0(best_seed_percentage, args.wave)
    final_y0_matrix = final_y0_flat.reshape(3, 6)

    print("\n--- Fitting Results ---")
    print(f"Success: {result.success}")
    print(f"Optimized Transmission Rate (beta): {best_beta:.4f}")
    print(f"Optimized Initial Seed Fraction: {best_seed_percentage * 100:.4f}%")
    print(f"Final Loss (RSS): {result.fun:.2f}")

    print("\n--- Optimized Initial Headcounts (t=0) ---")
    age_labels = ["0-20 (Young)", "21-64 (Middle)", "65+ (Elderly)"]
    for i in range(3):
        print(f"{age_labels[i]}: S={final_y0_matrix[i,0]:.0f}, E={final_y0_matrix[i,1]:.0f}, I={final_y0_matrix[i,2]:.0f}")

    # 5. Generate Fitted Model Curves for Plotting
    fitted_solution = run_model(final_y0_flat, t, best_beta, model_params)
    fitted_hospitalized = np.sum(fitted_solution.reshape(len(t), 3, 6)[:, :, 3], axis=1)

    # 6. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(dates, observed_critical, 'ro', label='Observed Critical Data', alpha=0.6, markersize=4)
    plt.plot(dates, fitted_hospitalized, 'b-', label=f'Fitted Model ($\\beta$={best_beta:.2f}, Seed={best_seed_percentage*100:.3f}%)', linewidth=2)
    
    plt.title(f"SEIHDR Joint Parameter Calibration - {wave_title}")
    plt.xlabel("Date")
    plt.ylabel("Total Critical Hospitalizations")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Dynamic image saving and timestamped history isolation
    plots_dir = Path(__file__).parent / 'fitted_plots'
    plots_dir.mkdir(exist_ok=True)
    
    archive_old_fitted_plots(plots_dir)
    plt.savefig(plots_dir / f"fitting_wave_{args.wave}.png")
    print(f"Saved fresh plot to {plots_dir / f'fitting_wave_{args.wave}.png'}")
    
    plt.show()

if __name__ == '__main__':
    main()