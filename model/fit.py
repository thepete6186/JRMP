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
# CONFIG
# ==============================================================================
WEIGHTED_START = np.datetime64("2022-01-01")
WEIGHTED_END = np.datetime64("2022-02-01")
WEIGHT_MULTIPLIER = 0.3

BETA_MIN = 0.2
BETA_MAX = 1.2  # Tightened upper bound to prevent explosive wave 4 growth
BETA_ROUGHNESS_WEIGHT = 5000.0  # Slightly increased to support smoothing
BETA_BOUNDARY_PENALTY = 1e5

SEED_BOUNDS = (1e-6, 0.02)
CONSTANT_BETA_BOUNDS = (0.05, 2.5)
SPLINE_BETA_BOUNDS = (0.1, 1.2)  # Tightened spline search space upper bound


def get_knot_times(t, n_knots=3):  # Restricted exclusively to 3 knots
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
):
    seed_percentage = params_to_fit[0]

    if use_splines and knot_times is not None:
        beta_knots = params_to_fit[1:]
        # Changed degree k=3 (cubic) to k=2 (quadratic spline)
        beta_func = make_interp_spline(knot_times, beta_knots, k=2)
        beta_eval = beta_func(t)

        # Penalize the entire evaluated timeline array instead of just sparse knots
        boundary_penalty = 0.0
        boundary_penalty += max(0.0, BETA_MIN - np.min(beta_eval)) ** 2
        boundary_penalty += max(0.0, np.max(beta_eval) - BETA_MAX) ** 2
        boundary_penalty *= BETA_BOUNDARY_PENALTY
    else:
        beta = params_to_fit[1] if len(params_to_fit) > 1 else params_to_fit[0]
        beta_eval = beta

        boundary_penalty = 0.0
        boundary_penalty += max(0.0, BETA_MIN - beta) ** 2
        boundary_penalty += max(0.0, beta - BETA_MAX) ** 2
        boundary_penalty *= BETA_BOUNDARY_PENALTY

    y0 = build_y0(seed_percentage, wave)
    y_hat = run_model(y0, t, beta_eval, static_params)
    state_reshaped = y_hat.reshape(len(t), 3, 6)
    model_total_critical = np.sum(state_reshaped[:, :, 3], axis=1)

    resid = observed_critical - model_total_critical

    if weights is None:
        loss = np.sum(resid ** 2)
    else:
        loss = np.sum(weights * resid ** 2)

    if use_splines:
        roughness = np.sum(np.diff(beta_knots, n=2) ** 2)
        loss += BETA_ROUGHNESS_WEIGHT * roughness

    return loss + boundary_penalty


def fit_with_multistart(
    t,
    observed_critical,
    model_params,
    wave,
    use_splines,
    knot_times,
    weights,
):
    seed_grid = np.logspace(np.log10(SEED_BOUNDS[0]), np.log10(SEED_BOUNDS[1]), 10)

    if use_splines:
        base_beta = 0.6
        # Initializing parameters specifically for 3 knots (1 seed parameter + 3 beta knots)
        x0_list = [[seed0] + [base_beta] * 3 for seed0 in seed_grid]
        bounds = [SEED_BOUNDS] + [SPLINE_BETA_BOUNDS] * 3
    else:
        x0_list = [[seed0, 0.5] for seed0 in seed_grid]
        bounds = [SEED_BOUNDS, CONSTANT_BETA_BOUNDS]

    best_result = None

    for x0 in x0_list:
        result = minimize(
            objective_function,
            x0,
            args=(t, observed_critical, model_params, wave, use_splines, knot_times, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9}
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parameter fitting with weighted wave 5 data.")
    parser.add_argument("--how", choices=["least_squares"], default="least_squares")
    parser.add_argument("--wave", choices=["4", "5"], default="4")
    parser.add_argument("--splines", choices=["y", "n"], default="n")
    args = parser.parse_args(argv)

    if args.wave == "4":
        end_date = "2021-05-15"
        model_params = ANCESTRAL_PARAMS
        wave_title = "Wave 4 (Ancestral)"
        start_dates = ["2020-10-15", "2020-11-01", "2020-11-15", "2020-12-01"]
    else:
        end_date = "2022-06-01"
        model_params = OMICRON_PARAMS
        wave_title = "Wave 5 (Omicron)"
        start_dates = ["2022-01-01", "2022-01-15", "2022-02-01", "2022-02-15"]

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
        knot_times = get_knot_times(t, n_knots=3) if use_splines else None
        weights = build_weights(np.array(dates, dtype="datetime64[D]"), args.wave)

        print(f"Optimizing {'3-knot quadratic splines' if use_splines else 'constant beta'} + seed percentage...")

        result = fit_with_multistart(
            t=t,
            observed_critical=observed_critical,
            model_params=model_params,
            wave=args.wave,
            use_splines=use_splines,
            knot_times=knot_times,
            weights=weights,
        )

        best_seed = result.x[0]

        if use_splines:
            best_beta_knots = result.x[1:]
            print(f"Optimized beta at 3 knots: {best_beta_knots.round(4)}")
            # Evaluation uses quadratic spline degree k=2
            beta_func = make_interp_spline(knot_times, best_beta_knots, k=2)
            beta_t = beta_func(t)
            fitted_solution = run_model(build_y0(best_seed, args.wave), t, beta_t, model_params)
        else:
            best_beta = result.x[1]
            fitted_solution = run_model(build_y0(best_seed, args.wave), t, best_beta, model_params)

        fitted_hospitalized = np.sum(fitted_solution.reshape(len(t), 3, 6)[:, :, 3], axis=1)

        print("\n--- Fitting Results ---")
        print(f"Start Date: {start_date}")
        print(f"Success: {result.success}")
        print(f"Optimized Initial Seed Fraction: {best_seed * 100:.6f}%")
        print(f"Final Loss (Weighted RSS): {result.fun:.2f}")

        n_params = len(result.x)
        n = len(observed_critical)
        aic = n * np.log(result.fun / n) + 2 * n_params if n > 0 and result.fun > 0 else float("inf")
        print(f"Approx AIC: {aic:.2f} (params={n_params})")

        plt.figure(figsize=(12, 6))
        plt.plot(dates, observed_critical, "ro", label="Observed Critical Data", alpha=0.6, markersize=4)
        if use_splines:
            label = f"Fitted (3-knot quadratic spline, Seed={best_seed*100:.3f}%)"
        else:
            label = f"Fitted Model ($\\beta$={best_beta:.2f}, Seed={best_seed*100:.3f}%)"
        plt.plot(dates, fitted_hospitalized, "b-", label=label, linewidth=2)

        plt.title(f"SEIHDR Calibration - {wave_title} ({start_date} to {end_date})")
        plt.xlabel("Date")
        plt.ylabel("Total Critical Hospitalizations")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        suffix = "_splines_quad3" if use_splines else ""
        if use_splines:
            save_path = splineplots_dir / f"fitting_wave_{args.wave}_{start_date}{suffix}.png"
        else:
            save_path = plots_dir / f"fitting_wave_{args.wave}_{start_date}{suffix}.png"

        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}\n")
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()