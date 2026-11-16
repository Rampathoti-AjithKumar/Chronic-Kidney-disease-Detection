import numpy as np


# ---------------------------------------------------------
# Quality-Aware Federated Aggregation
# ---------------------------------------------------------

def aggregate_parameters(
    client_parameters,
    client_sample_counts,
    quality_scores
):

    total_weight = 0.0

    # Calculate quality-adjusted weights
    adjusted_weights = []

    for sample_count, quality_score in zip(
        client_sample_counts,
        quality_scores
    ):

        weight = sample_count * quality_score

        adjusted_weights.append(weight)

        total_weight += weight

    # Avoid division by zero
    if total_weight == 0:

        raise ValueError(
            "Total aggregation weight cannot be zero."
        )

    # Normalize weights
    normalized_weights = [
        weight / total_weight
        for weight in adjusted_weights
    ]

    # Number of parameter arrays
    num_parameters = len(
        client_parameters[0]
    )

    aggregated_parameters = []

    # Aggregate each parameter
    for parameter_index in range(
        num_parameters
    ):

        aggregated_parameter = np.zeros_like(
            client_parameters[0][parameter_index]
        )

        for client_index in range(
            len(client_parameters)
        ):

            aggregated_parameter += (
                normalized_weights[client_index]
                * client_parameters[client_index][parameter_index]
            )

        aggregated_parameters.append(
            aggregated_parameter
        )

    return aggregated_parameters, normalized_weights