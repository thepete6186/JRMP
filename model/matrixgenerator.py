# This script aggregates the original 15x15 age-contact matrix into 3 main age groups.
import numpy as np
import shutil

original_matrix = np.array([
    [3.11, 0.51, 0.14, 0.07, 0.67, 0.86, 1.58, 0.85, 0.71, 0.21, 0.34, 0.50, 0.46, 0.13, 0.05],
    [0.43, 3.50, 0.77, 0.27, 0.22, 0.48, 0.86, 0.90, 0.98, 0.48, 0.15, 0.06, 0.22, 0.17, 0.27],
    [0.05, 0.42, 6.56, 0.37, 0.47, 0.28, 0.61, 0.72, 1.18, 0.73, 0.61, 0.06, 0.11, 0.07, 0.37],
    [0.02, 0.05, 0.49, 2.35, 1.47, 0.23, 0.69, 0.23, 0.63, 0.71, 0.79, 0.16, 0.05, 0.00, 0.04],
    [0.08, 0.07, 0.16, 0.53, 2.16, 0.95, 1.54, 0.68, 1.37, 0.45, 1.04, 0.89, 0.49, 0.02, 0.00],
    [0.16, 0.02, 0.01, 0.59, 0.45, 1.43, 0.76, 0.38, 0.65, 0.32, 0.17, 0.21, 0.40, 0.23, 0.16],
    [0.38, 0.28, 0.84, 0.18, 0.80, 1.03, 2.32, 0.58, 0.92, 0.49, 0.51, 0.34, 0.79, 0.12, 0.23],
    [0.45, 0.31, 0.23, 0.14, 0.23, 0.40, 0.78, 0.86, 1.08, 0.33, 0.33, 0.17, 0.28, 0.13, 0.17],
    [0.12, 0.23, 0.42, 0.13, 0.32, 0.64, 0.76, 0.65, 1.15, 0.86, 0.70, 0.31, 0.22, 0.05, 0.08],
    [0.06, 0.19, 0.30, 0.38, 0.41, 0.62, 0.99, 0.84, 1.14, 0.98, 0.82, 0.50, 0.53, 0.01, 0.14],
    [0.07, 0.14, 0.26, 0.34, 0.49, 0.55, 0.84, 0.47, 0.77, 1.25, 0.97, 0.51, 0.33, 0.07, 0.20],
    [0.47, 0.06, 0.08, 0.16, 0.41, 0.51, 0.66, 0.63, 0.82, 0.70, 1.18, 0.73, 0.65, 0.41, 0.31],
    [0.10, 0.03, 0.04, 0.04, 0.23, 0.36, 0.40, 0.38, 0.30, 0.42, 0.51, 0.80, 0.61, 0.35, 0.18],
    [0.21, 0.00, 0.09, 0.05, 0.23, 0.34, 0.41, 0.51, 0.57, 0.19, 0.32, 0.39, 0.67, 0.26, 0.18],
    [0.03, 0.02, 0.00, 0.22, 0.16, 0.31, 0.38, 0.54, 0.54, 0.26, 0.19, 0.33, 0.55, 0.44, 0.66]
], dtype=float)
#Original matrix from the paper 

populationcount = [213.1, 270.9, 290.1, 268.8, 302.9, 436.1, 512.2, 561.0, 584.7, 566.6, 576.8, 605.9, 628.1, 523.5, 395.0+221.1+159.3+230.0]


def aggregate_contact_matrix(matrix, population, groups):
    """
    Aggregate an age-specific contact matrix into larger age groups.

    The aggregation is asymmetric: for each source group, contacts to all
    destination ages in the target group are summed, then averaged across the
    source ages using population weights.

    Parameters
    ----------
    matrix : np.ndarray
        Square contact matrix for the fine age groups.
    population : array-like
        Population counts for each fine age group.
    groups : list[list[int]]
        Indices for the fine age groups that belong to each coarse group.

    Returns
    -------
    np.ndarray
        Aggregated contact matrix.
    """
    matrix = np.asarray(matrix, dtype=float)
    population = np.asarray(population, dtype=float)

    aggregated = np.zeros((len(groups), len(groups)), dtype=float)

    for i, source_group in enumerate(groups):
        source_weights = population[source_group]
        for j, target_group in enumerate(groups):
            source_to_target = matrix[np.ix_(source_group, target_group)].sum(axis=1)
            aggregated[i, j] = np.average(source_to_target, weights=source_weights)

    return aggregated


age_groups_3 = [
    list(range(0, 4)),   # 0-20
    list(range(4, 13)),  # 21-64
    list(range(13, 15)), # 65+
]


if __name__ == "__main__":
    population_array = np.asarray(populationcount, dtype=float)
    contact_3x3 = aggregate_contact_matrix(original_matrix, population_array, age_groups_3)

    np.set_printoptions(precision=4, suppress=True)
    print("3x3 aggregated contact matrix (0-20, 21-64, 65+):")
    print(contact_3x3)


final_matrix = np.array([[4.8821, 4.998, 0.2838],
                         [0.8814, 5.9716, 0.3371],
                         [0.2974, 3.3867, 0.874]], dtype=float)





