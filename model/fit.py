import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline


# ==============================================================================
# 1. FIX PATH MANIPULATION BEFORE ANY CUSTOM IMPORTS
# ==============================================================================
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))


from SEIHDR import run_model, ANCESTRAL_PARAMS, OMICRON_PARAMS, DEFAULT_POPULATION
from data.plot import load_hkgov_critical_series


# ==============================================================================
# CONFIG & CALIBRATION BOUNDS
# ==============================================================================
WEIGHTED_START = np.datetime64("2022-02-01")
WEIGHTED_END = np.datetime64("2022-03-01")
WEIGHT_MULTIPLIER = 0.3

BETA_MIN = 0.001
BETA_MAX = 5.0                  
BETA_ROUGHNESS_WEIGHT = 1000.0  
BETA_BOUNDARY_PENALTY = 1e5

# Co-optimization search space boundaries
SEED_BOUNDS = (1e-5, 0.05)       
CONSTANT_BETA_BOUNDS = (0.001, 5.0)
SPLINE_BETA_BOUNDS = (0.001, 5.0) 


def calculate_r0(beta_array, static_params):
    # Assuming typical SEIR progression: R0 = beta * infectious_duration
    infectious_period = 4.5  
    return beta_array * infectious_period


def get_knot_times(t, observed_critical, n_knots=3):
    if n_knots == 3:
        cum_data = np.cumsum(observed_critical)
        if cum_data[-1] == 0:
            return np.linspace(0, len(t) - 1, n_knots, dtype=int)
        
        mid_idx = np.searchsorted(cum_data, cum_data[-1] * 0.5)
        knots = [0, int(mid_idx), len(t) - 1]
        
        if knots[1] == knots[0]:
            knots[1] = len(t) // 2
        return np.array(knots)
    else:
        return np.linspace(0, len(t) - 1, n_knots, dtype=int)


def build_y0(seed_percentage, wave):
    y0_matrix = np.zeros((3, 6))
    for i in range(3):
        if wave == "5":
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


def build_weights(dates, wave):
    weights = np.ones(len(dates), dtype=float)
    if wave == "5":
        mask = (dates >= WEIGHTED_START) & (dates < WEIGHTED_END)
        weights[mask] = WEIGHT_MULTIPLIER
    return weights


def objective_function(
    params_to_fit,
    t,
    observed_critical,
    static_params,
    wave,
    use_splines=False,
    knot_times=None,
    weights=None,
    how="rss-rough",
):
    seed_percentage = params_to_fit[0]

    if use_splines and knot_times is not None:
        beta_knots = params_to_fit[1:]
        beta_func = make_interp_spline(knot_times, beta_knots, k=1)
        beta_eval = beta_func(t)
    else:
        beta = params_to_fit[1]
        beta_eval = np.full(len(t), beta)

    boundary_penalty = 0.0
    boundary_penalty += max(0.0, BETA_MIN - np.min(beta_eval)) ** 2
    boundary_penalty += max(0.0, np.max(beta_eval) - BETA_MAX) ** 2
    boundary_penalty *= BETA_BOUNDARY_PENALTY

    y0 = build_y0(seed_percentage, wave)
    y_hat = run_model(y0, t, beta_eval, static_params)
    state_reshaped = y_hat.reshape(len(t), 3, 6)
    model_total_critical = np.sum(state_reshaped[:, :, 3], axis=1)

    resid = observed_critical - model_total_critical
    if weights is not None:
        resid = weights * resid

    loss = np.sum(resid ** 2)
        
    if use_splines:
        if how == "rss-rough":
            # Smooths out abrupt slope changes
            roughness = np.mean(np.diff(beta_eval, n=1) ** 2)
            loss += BETA_ROUGHNESS_WEIGHT * roughness
            
        # DEFENSE FIX: Epidemic End Anchor
        # If the curve goes up at the very end when data is low, penalize it heavily.
        # We check if the final knot value is greater than the middle knot value.
        if len(beta_knots) == 3:
            final_slope = beta_knots[2] - beta_knots[1]
            if final_slope > 0:
                # Severe penalty for an uncharacteristic late-wave explosion
                loss += 5e4 * (final_slope ** 2)

    return loss + boundary_penalty


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parameter fitting with joint optimization loops.")
    parser.add_argument("--how", choices=["rss", "rss-rough"], default="rss-rough")
    parser.add_argument("--wave", choices=["4", "5"], default="4")
    parser.add_argument("--splines", choices=["y", "n"], default="n")
    args = parser.parse_args(argv)

    if args.wave == "4":
        end_date = "2021-05-15"
        model_params = ANCESTRAL_PARAMS
        wave_title = "Wave 4 (Ancestral)"
        start_dates = ["2020-10-15", "2020-11-01", "2020-11-15", "2020-12-01"]
    else:
        end_date = "2022-05-10"
        model_params = OMICRON_PARAMS
        wave_title = "Wave 5 (Omicron)"
        start_dates = ["2022-01-15", "2022-02-01", "2022-02-15"]

    plots_dir = Path(__file__).parent / "fitted_plots"
    plots_dir.mkdir(exist_ok=True)
    splineplots_dir = Path(__file__).parent / "fitted_plots_splines"
    splineplots_dir.mkdir(exist_ok=True)

    for start_date in start_dates:
        print(f"Loading HK Gov data for {wave_title} from {start_date}...")
        dates, observed_critical = load_hkgov_critical_series(start_date=start_date, end_date=end_date)
        t = np.arange(len(dates))

        if len(observed_critical) == 0:
            print(f"Skipping {start_date}: no data returned.")
            continue

        use_splines = args.splines == "y"
        knot_times = get_knot_times(t, observed_critical, n_knots=3) if use_splines else None
        weights = build_weights(np.array(dates, dtype="datetime64[D]"), args.wave)

        print(f"Co-optimizing transmission vector AND initial seed percentage together...")

        initial_seed_guess = 0.001
        initial_beta_guess = 0.5

        if use_splines:
            x0 = [initial_seed_guess] + [initial_beta_guess] * 3
            bounds = [SEED_BOUNDS] + [SPLINE_BETA_BOUNDS] * 3
        else:
            x0 = [initial_seed_guess, initial_beta_guess]
            bounds = [SEED_BOUNDS, CONSTANT_BETA_BOUNDS]

        result = minimize(
            objective_function,
            x0,
            args=(t, observed_critical, model_params, args.wave, use_splines, knot_times, weights, args.how),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-7}
        )

        best_seed = result.x[0]

        if use_splines:
            best_beta_knots = result.x[1:]
            print(f"Optimized beta at adaptive knots: {best_beta_knots.round(4)}")
            # LINEAR SPLINE EVALUATION (k=1)
            beta_func = make_interp_spline(knot_times, best_beta_knots, k=1)
            beta_t = beta_func(t)
            fitted_solution = run_model(build_y0(best_seed, args.wave), t, beta_t, model_params)
            r0_trajectory = calculate_r0(beta_t, model_params)
        else:
            best_beta = result.x[1]
            fitted_solution = run_model(build_y0(best_seed, args.wave), t, best_beta, model_params)
            r0_trajectory = calculate_r0(np.full(len(t), best_beta), model_params)

        fitted_hospitalized = np.sum(fitted_solution.reshape(len(t), 3, 6)[:, :, 3], axis=1)

        print("\n--- Calibration Outputs ---")
        print(f"Success Status: {result.success}")
        print(f"Joint Seed Fraction Output: {best_seed * 100:.6f}%")
        print(f"Mean R0 Value across timeline: {np.mean(r0_trajectory):.2f}")
        print(f"Final Objective Function Loss Value: {result.fun:.2f}")

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Critical Hospitalizations', color=color)
        ax1.plot(dates, observed_critical, "ro", label="Observed Critical Data", alpha=0.5, markersize=4)
        
        if use_splines:
            case_label = f"Fitted Model (3-Knot Linear Splines, Seed={best_seed*100:.4f}%)"
        else:
            case_label = f"Fitted Model (Constant, $\\beta$={best_beta:.2f}, Seed={best_seed*100:.4f}%)"
            
        ax1.plot(dates, fitted_hospitalized, "b-", label=case_label, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.3)

        ax2 = ax1.twinx()  
        color = 'tab:green'
        ax2.set_ylabel('Calculated Reproduction Metric (R0)', color=color)
        ax2.plot(dates, r0_trajectory, 'g--', label=f'R0 Trajectory (Mean={np.mean(r0_trajectory):.2f})', linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"Joint SEIHDR Calibration with R0 Output - {wave_title} ({start_date})")
        fig.tight_layout()
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.xticks(rotation=45)

        suffix = f"_{args.how}_joint_linear_splines" if use_splines else f"_{args.how}_joint_constant"
        if use_splines:
            save_path = splineplots_dir / f"fitting_wave_{args.wave}_{start_date}{suffix}.png"
        else:
            save_path = plots_dir / f"fitting_wave_{args.wave}_{start_date}{suffix}.png"

        plt.savefig(save_path, dpi=200)
        print(f"Saved execution graph to {save_path}\n")
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()