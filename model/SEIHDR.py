import numpy as np
from scipy.integrate import odeint

from matrixgenerator import final_matrix, population_count

AGE_GROUPS = ("0-20", "21-64", "65+")
CONTACT_MATRIX = final_matrix
AGE_GROUP_SLICES = (slice(0, 4), slice(4, 13), slice(13, 15))

DEFAULT_POPULATION = np.array(
    [sum(population_count[s]) for s in AGE_GROUP_SLICES],
    dtype=float,
)

# Omicron variant (wave 5) parameters.
OMICRON_PARAMS = {
    "sigma": np.array([1/4, 1/4, 1/4], dtype=float),
    "tau1": np.array([1/6, 1/6, 1/6], dtype=float),
    "delta1": np.array([0.0, 0.0, 0.0], dtype=float),
    "tau2": np.array([0.152, 0.124, 0.091], dtype=float),
    "delta2": np.array([0.00154, 0.0299, 0.063], dtype=float),
    "alpha": np.array([0.0029, 0.0043, 0.012], dtype=float),
    "upsilon": np.array([1/160, 1/160, 1/160], dtype=float),
}

# Ancestral strain (wave 4) parameters.
ANCESTRAL_PARAMS = {
    "sigma": np.array([1/5.5, 1/5.5, 1/5.5], dtype=float),
    "tau1": np.array([1/6, 1/6, 1/6], dtype=float),
    "delta1": np.array([0.0, 0.0, 0.0], dtype=float),
    "tau2": np.array([0.083, 0.0817, 0.0658], dtype=float),
    "delta2": np.array([0.0, 0.00164, 0.0174], dtype=float),
    "alpha": np.array([0.0043, 0.019, 0.071], dtype=float),
    "upsilon": np.array([0.0, 0.0, 0.0], dtype=float),
}

def SEIHDR_model(
    y,
    t,
    beta,  # float, callable, or array matching the time grid
    sigma,
    tau1,
    tau2,
    alpha,
    delta1,
    delta2,
    upsilon,
    contact_matrix=CONTACT_MATRIX,
    population=None,
    t_grid=None,  # required when beta is a pre-computed array
):
    """Age-stratified SEIHDR model with constant, functional, or array beta."""
    state = np.asarray(y, dtype=float).reshape(len(AGE_GROUPS), 6)

    susceptible = state[:, 0]
    exposed = state[:, 1]
    infected = state[:, 2]
    hospitalized = state[:, 3]
    recovered = state[:, 4]
    dead = state[:, 5]

    if population is None:
        population = DEFAULT_POPULATION
    else:
        population = np.asarray(population, dtype=float)

    if callable(beta):
        current_beta = beta(t)
    elif isinstance(beta, np.ndarray):
        if t_grid is None:
            raise ValueError("If beta is an array, you must pass t_grid to SEIHDR_model.")
        current_beta = np.interp(t, t_grid, beta)
    else:
        current_beta = beta

    sigma = np.asarray(sigma)
    tau1 = np.asarray(tau1)
    tau2 = np.asarray(tau2)
    alpha = np.asarray(alpha)
    delta1 = np.asarray(delta1)
    delta2 = np.asarray(delta2)
    upsilon = np.asarray(upsilon)

    infectious_fraction = np.divide(
        infected + 0.1 * hospitalized,
        population,
        out=np.zeros_like(infected, dtype=float),
        where=population > 0,
    )

    lambda_k = current_beta * np.asarray(contact_matrix, dtype=float).dot(infectious_fraction)

    dSdt = -lambda_k * susceptible + upsilon * recovered
    dEdt = lambda_k * susceptible - sigma * exposed
    dIdt = sigma * exposed - (tau1 + alpha + delta1) * infected
    dHdt = alpha * infected - (tau2 + delta2) * hospitalized
    dRdt = tau1 * infected + tau2 * hospitalized - upsilon * recovered
    dDdt = delta1 * infected + delta2 * hospitalized

    derivatives = np.column_stack([dSdt, dEdt, dIdt, dHdt, dRdt, dDdt])
    return derivatives.reshape(-1)


def run_model(y0, t, beta, params):
    """Run the model over time grid t from initial state y0."""
    return odeint(
        SEIHDR_model,
        y0,
        t,
        args=(
            beta,
            params["sigma"],
            params["tau1"],
            params["tau2"],
            params["alpha"],
            params["delta1"],
            params["delta2"],
            params["upsilon"],
            CONTACT_MATRIX,
            DEFAULT_POPULATION,
            t,
        ),
    )