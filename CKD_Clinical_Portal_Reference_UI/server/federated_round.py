from server.quality_evaluator import get_performance_score
from server.stability_evaluator import calculate_stability
from server.compatibility_evaluator import calculate_compatibility
from server.quality_score import calculate_quality_score
from server.aggregation import aggregate_parameters


def run_federated_round(
    clients,
    reference_data,
    previous_parameters=None
):

    print("\n===================================")
    print("       FEDERATED TRAINING ROUND")
    print("===================================")

    client_parameters = []
    client_sample_counts = []
    quality_scores = []

    # Store current client parameters
    current_parameters = []

    # ------------------------------------------------
    # 1. Collect parameters from trained clients
    # ------------------------------------------------

    for client in clients:

        print(
            f"\nCollecting parameters from "
            f"{client.client_id}..."
        )

        parameters = client.get_parameters()

        current_parameters.append(
            parameters
        )

        client_parameters.append(
            parameters
        )

        client_sample_counts.append(
            client.get_num_samples()
        )

    # ------------------------------------------------
    # 2. Evaluate each client
    # ------------------------------------------------

    for index, client in enumerate(clients):

        performance, metrics = get_performance_score(
            client.model,
            client.X_validation,
            client.y_validation
        )

        # --------------------------------------------
        # Stability
        # --------------------------------------------

        if previous_parameters is None:

            # First federated round
            # No previous parameters available
            stability = 1.0

        else:

            stability = calculate_stability(
                previous_parameters[index],
                current_parameters[index]
            )

        # --------------------------------------------
        # Compatibility
        # --------------------------------------------

        compatibility = calculate_compatibility(
            client.X_train,
            reference_data
        )

        # --------------------------------------------
        # Quality Score
        # --------------------------------------------

        quality = calculate_quality_score(
            performance,
            stability,
            compatibility
        )

        quality_scores.append(
            quality
        )

        print(
            f"\n{client.client_id} Quality Evaluation:"
        )

        print(
            f"  Performance: "
            f"{performance:.4f}"
        )

        print(
            f"  Stability: "
            f"{stability:.4f}"
        )

        print(
            f"  Compatibility: "
            f"{compatibility:.4f}"
        )

        print(
            f"  Quality Score: "
            f"{quality:.4f}"
        )

    # ------------------------------------------------
    # 3. Quality-Aware Aggregation
    # ------------------------------------------------

    global_parameters, normalized_weights = (
        aggregate_parameters(
            client_parameters,
            client_sample_counts,
            quality_scores
        )
    )

    print(
        "\nQuality-Aware Aggregation:"
    )

    for index, client in enumerate(clients):

        print(
            f"  {client.client_id}: "
            f"{normalized_weights[index]:.4f}"
        )

    print(
        "\nGlobal model parameters "
        "generated successfully."
    )

    # ------------------------------------------------
    # 4. Return results
    # ------------------------------------------------

    return (
        global_parameters,
        current_parameters,
        quality_scores
    )