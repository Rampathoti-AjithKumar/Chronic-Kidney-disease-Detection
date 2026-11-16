import numpy as np


# ---------------------------------------------------------
# 1. Calculate parameter change
# ---------------------------------------------------------

def calculate_parameter_change(
    previous_parameters,
    current_parameters
):

    total_change = 0.0

    for previous, current in zip(
        previous_parameters,
        current_parameters
    ):

        difference = current - previous

        total_change += np.linalg.norm(
            difference
        )

    return total_change


# ---------------------------------------------------------
# 2. Calculate stability score
# ---------------------------------------------------------

def calculate_stability(
    previous_parameters,
    current_parameters
):

    parameter_change = calculate_parameter_change(
        previous_parameters,
        current_parameters
    )

    # Convert parameter change into stability score
    stability_score = 1 / (
        1 + parameter_change
    )

    return stability_score