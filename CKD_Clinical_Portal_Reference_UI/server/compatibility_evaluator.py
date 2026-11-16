import numpy as np


# ---------------------------------------------------------
# 1. Calculate distribution difference
# ---------------------------------------------------------

def calculate_distribution_difference(
    client_data,
    reference_data
):

    client_mean = np.mean(
        client_data,
        axis=0
    )

    reference_mean = np.mean(
        reference_data,
        axis=0
    )

    difference = np.linalg.norm(
        client_mean - reference_mean
    )

    return difference


# ---------------------------------------------------------
# 2. Calculate compatibility score
# ---------------------------------------------------------

def calculate_compatibility(
    client_data,
    reference_data
):

    difference = calculate_distribution_difference(
        client_data,
        reference_data
    )

    compatibility_score = 1 / (
        1 + difference
    )

    return compatibility_score