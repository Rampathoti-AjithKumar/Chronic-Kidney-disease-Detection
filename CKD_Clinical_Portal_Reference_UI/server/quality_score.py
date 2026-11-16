# ---------------------------------------------------------
# Quality Score Calculation
# ---------------------------------------------------------

def calculate_quality_score(
    performance,
    stability,
    compatibility,
    performance_weight=0.50,
    stability_weight=0.25,
    compatibility_weight=0.25
):

    # Check that weights add up to 1
    total_weight = (
        performance_weight
        + stability_weight
        + compatibility_weight
    )

    if total_weight != 1.0:
        raise ValueError(
            "Quality score weights must add up to 1."
        )

    # Calculate weighted quality score
    quality_score = (
        performance_weight * performance
        + stability_weight * stability
        + compatibility_weight * compatibility
    )

    return quality_score