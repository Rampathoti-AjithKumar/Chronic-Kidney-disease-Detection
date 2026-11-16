import numpy as np
import pandas as pd
import shap


def explain_prediction(
    model,
    X_background,
    X_patient,
    feature_names
):
    """
    Generate SHAP explanation for a CKD prediction.

    The model internally treats:
        0 = CKD
        1 = notCKD

    SHAP LinearExplainer explains the model output toward
    class 1 (notCKD). Therefore, we negate the SHAP values
    to obtain a CKD-oriented explanation.
    """

    X_background = pd.DataFrame(
        X_background,
        columns=feature_names
    )

    X_patient = pd.DataFrame(
        X_patient,
        columns=feature_names
    )

    # SHAP background data
    explainer = shap.LinearExplainer(
        model,
        X_background
    )

    shap_values = explainer(X_patient)

    return explainer, shap_values


def display_explanation(
    model,
    X_background,
    X_patient,
    feature_names,
    target_encoder
):
    """
    Display a CKD-oriented SHAP explanation.
    """

    explainer, shap_values = explain_prediction(
        model,
        X_background,
        X_patient,
        feature_names
    )

    # Convert DataFrame to NumPy before passing to sklearn.
    # This removes feature-name warnings.
    X_patient_numpy = X_patient.to_numpy()

    prediction_encoded = model.predict(
        X_patient_numpy
    )[0]

    prediction_label = target_encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    probabilities = model.predict_proba(
        X_patient_numpy
    )[0]

    classes = model.classes_

    print("\n===================================")
    print("          XAI EXPLANATION")
    print("===================================")

    print(f"\nPrediction: {prediction_label}")

    print("\nClass Probabilities:")

    for class_value, probability in zip(
        classes,
        probabilities
    ):
        class_label = target_encoder.inverse_transform(
            [class_value]
        )[0]

        print(
            f"  {class_label}: {probability:.4f}"
        )

    # ------------------------------------------------
    # SHAP DIRECTION
    # ------------------------------------------------
    #
    # LinearExplainer explains class 1 = notCKD.
    #
    # Therefore:
    #
    # Positive raw SHAP  -> supports notCKD
    # Negative raw SHAP  -> supports CKD
    #
    # For our project we want CKD-oriented SHAP.
    # Therefore we reverse the sign.
    # ------------------------------------------------

    raw_shap_values = shap_values.values[0]

    ckd_shap_values = -raw_shap_values

    feature_values = X_patient.iloc[0].values

    explanation_data = []

    for feature, value, shap_value in zip(
        feature_names,
        feature_values,
        ckd_shap_values
    ):
        explanation_data.append(
            (
                feature,
                value,
                shap_value
            )
        )

    # Sort by absolute contribution
    explanation_data.sort(
        key=lambda x: abs(x[2]),
        reverse=True
    )

    print("\nTop Feature Contributions:")
    print("(CKD-oriented SHAP explanation)")

    for feature, value, shap_value in explanation_data[:10]:

        if shap_value > 0:
            direction = "supports CKD"
        elif shap_value < 0:
            direction = "supports notCKD"
        else:
            direction = "no significant contribution"

        print(
            f"  {feature}: "
            f"value={value:.4f}, "
            f"SHAP={shap_value:.4f} "
            f"({direction})"
        )

    return {
        "prediction": prediction_label,
        "probabilities": probabilities,
        "shap_values": ckd_shap_values,
        "feature_names": feature_names,
        "feature_values": feature_values
    }


if __name__ == "__main__":
    print("SHAP XAI module loaded successfully.")