import numpy as np
from scipy.integrate import odeint

from matrixgenerator import final_matrix, populationcount


AGE_GROUPS = ("0-20", "21-64", "65+")
CONTACT_MATRIX = final_matrix
AGE_GROUP_SLICES = (slice(0, 4), slice(4, 13), slice(13, 15))

DEFAULT_POPULATION = np.array(
    [sum(populationcount[s]) for s in AGE_GROUP_SLICES],
    dtype=float,
)


#Parameters here

OMICRON_PARAMS = {
    "sigma": np.array([1/4, 1/4, 1/4], dtype=float),

    "tau1": np.array([1/6, 1/6, 1/6], dtype=float),

    "delta1": np.array([0.0, 0.0, 0.0], dtype=float),  #I decided to not have any deaths but I was too lazy to remove delta.

    "tau2": np.array([0.152, 0.124, 0.091], dtype=float),

    "delta2": np.array([0.00154, 0.0299, 0.063], dtype=float),

    "alpha": np.array([0.0029, 0.0043, 0.012], dtype=float),

    "upsilon": np.array([1/160, 1/160, 1/160], dtype=float),
}


# ---- ANCESTRAL ----
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
    beta,
    sigma,
    tau1,
    tau2,
    alpha,
    delta1,
    delta2,
    upsilon,
    contact_matrix=CONTACT_MATRIX,
    population=None,
):
    """
    Age-stratified SEIHDR model (3 age groups × 6 compartments).
    """

    # reshape state
    state = np.asarray(y, dtype=float).reshape(len(AGE_GROUPS), 6)

    susceptible = state[:, 0]
    exposed = state[:, 1]
    infected = state[:, 2]
    hospitalized = state[:, 3]
    recovered = state[:, 4]
    dead = state[:, 5]

    # population
    if population is None:
        population = DEFAULT_POPULATION
    else:
        population = np.asarray(population, dtype=float)

    # force arrays (prevent broadcasting bugs)
    sigma = np.asarray(sigma)
    tau1 = np.asarray(tau1)
    tau2 = np.asarray(tau2)
    alpha = np.asarray(alpha)
    delta1 = np.asarray(delta1)
    delta2 = np.asarray(delta2)
    upsilon = np.asarray(upsilon)

    # force of infection
    infectious_fraction = np.divide(
        infected + 0.1* hospitalized,  #assume hospitzlied are less infectious
        population,
        out=np.zeros_like(infected, dtype=float),
        where=population > 0,
    )

    lambda_k = beta * np.asarray(contact_matrix, dtype=float).dot(infectious_fraction)

    # differential equations
    dSdt = -lambda_k * susceptible + upsilon * recovered
    dEdt = lambda_k * susceptible - sigma * exposed
    dIdt = sigma * exposed - (tau1 + alpha + delta1) * infected
    dHdt = alpha * infected - (tau2 + delta2) * hospitalized
    dRdt = tau1 * infected + tau2 * hospitalized - upsilon * recovered
    dDdt = delta1 * infected + delta2 * hospitalized

    derivatives = np.column_stack([dSdt, dEdt, dIdt, dHdt, dRdt, dDdt])
    return derivatives.reshape(-1)



def run_model(y0, t, beta, params):
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
        ),
    )

