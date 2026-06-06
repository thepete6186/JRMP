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
    """Age-stratified SEIHDR model using the 3x3 contact matrix from matrixgenerator.

    Parameters
    ----------
    y : array-like
        Flattened state vector with 6 compartments per age group.
    t : float
        Time.
    beta, mu, sigma, tau1, tau2, alpha, delta1, delta2, upsilon : float
        Model parameters. 
    contact_matrix : np.ndarray, optional
        3x3 asymmetric contact matrix. Defaults to matrixgenerator.final_matrix.
    population : array-like, optional
        Population size per age group. If omitted, the current compartment totals
        are used.
    """
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

    infectious_fraction = np.divide(
        infected + hospitalized,
        population,
        out=np.zeros_like(infected, dtype=float),
        where=population > 0,
    )
    force_of_infection = beta * np.asarray(contact_matrix, dtype=float).dot(infectious_fraction)

    dSdt = -force_of_infection * susceptible+ upsilon * recovered
    dEdt = force_of_infection * susceptible - sigma * exposed
    dIdt = sigma * exposed - (tau1 + alpha + delta1) * infected
    dHdt = alpha * infected - (tau2 + delta2) * hospitalized
    dRdt = tau1 * infected + tau2 * hospitalized - upsilon * recovered
    dDdt = delta1 * infected + delta2 * hospitalized

    derivatives = np.column_stack([dSdt, dEdt, dIdt, dHdt, dRdt, dDdt])
    return derivatives.reshape(-1)