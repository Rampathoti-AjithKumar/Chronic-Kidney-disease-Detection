import numpy as np
from preprocessing.preprocess import prepare_data

from clients.client_data import (
    create_client_datasets,
    show_client_distribution
)

from clients.client import CKDClient

from models.ckd_model import (
    create_model,
    set_model_parameters,
    evaluate_model
)

from server.federated_round import (
    run_federated_round
)



# --------------------------------------------------
# 1. Prepare Dataset
# --------------------------------------------------

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    label_encoders,
    target_encoder
) = prepare_data()


# --------------------------------------------------
# 2. Create Client Datasets
# --------------------------------------------------

clients_data = create_client_datasets(
    X_train,
    y_train,
    num_clients=3
)

show_client_distribution(
    clients_data
)


# --------------------------------------------------
# 3. Create and Train Local Clients
# --------------------------------------------------

clients = []

for client_id, data in clients_data.items():

    client = CKDClient(
        client_id,
        data["X_train"],
        data["y_train"],
        data["X_validation"],
        data["y_validation"]
    )

    # Local training
    client.train()

    # Local validation
    metrics = client.evaluate()

    print(
        f"\n{client_id} Validation Metrics:"
    )

    print(
        f"  Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"  F1 Score: "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"  ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    clients.append(client)


# --------------------------------------------------
# 4. Display Local Model Information
# --------------------------------------------------

print(
    "\nLocal Client Training Summary:"
)

for client in clients:

    parameters = client.get_parameters()

    print(
        f"\n{client.client_id}: "
        f"{client.get_num_samples()} training samples"
    )

    print(
        f"  Weight shape: "
        f"{parameters[0].shape}"
    )

    print(
        f"  Bias shape: "
        f"{parameters[1].shape}"
    )


# --------------------------------------------------
# 5. Federated Training Round
# --------------------------------------------------

print(
    "\nStarting Federated Learning..."
)

reference_data = X_train

(
    global_parameters,
    current_parameters,
    quality_scores
) = run_federated_round(
    clients,
    reference_data
)


# --------------------------------------------------
# 6. Display Federated Learning Results
# --------------------------------------------------

print(
    "\nFederated Training Result:"
)

for i, client in enumerate(clients):

    print(
        f"{client.client_id} "
        f"Quality Score: "
        f"{quality_scores[i]:.4f}"
    )

print(
    "\nGlobal model parameters "
    "generated successfully."
)


# --------------------------------------------------
# 7. Create Global Model
# --------------------------------------------------

print(
    "\nCreating Global CKD Model..."
)

global_model = create_model()

global_model = set_model_parameters(
    global_model,
    global_parameters
)

print(
    "Global model created successfully."
)


# --------------------------------------------------
# 8. Evaluate Global Model
# --------------------------------------------------

print(
    "\n===================================")
print(
    "       GLOBAL MODEL EVALUATION"
)
print(
    "==================================="
)

global_metrics = evaluate_model(
    global_model,
    X_test,
    y_test
)


print(
    f"Global Accuracy: "
    f"{global_metrics['accuracy']:.4f}"
)

print(
    f"Global Precision: "
    f"{global_metrics['precision']:.4f}"
)

print(
    f"Global Recall: "
    f"{global_metrics['recall']:.4f}"
)

print(
    f"Global F1 Score: "
    f"{global_metrics['f1_score']:.4f}"
)

print(
    f"Global ROC-AUC: "
    f"{global_metrics['roc_auc']:.4f}"
)

# Inspect probability extremes on the untouched test set. These are diagnostics,
# not a reason to manually alter the reported probabilities.
test_probabilities = global_model.predict_proba(X_test.to_numpy())
ckd_column = list(global_model.classes_).index(0)
ckd_test_probabilities = test_probabilities[:, ckd_column]
print("\nTest-set CKD probability diagnostics:")
print(f"  Minimum: {ckd_test_probabilities.min():.8f}")
print(f"  Median:  {np.median(ckd_test_probabilities):.8f}")
print(f"  Maximum: {ckd_test_probabilities.max():.8f}")
print(f"  Probabilities < 0.001 or > 0.999: "
      f"{np.sum((ckd_test_probabilities < 0.001) | (ckd_test_probabilities > 0.999))} / "
      f"{len(ckd_test_probabilities)}")


# --------------------------------------------------
# 9. Global Test Dataset Information
# --------------------------------------------------

print(
    "\nGlobal Test Dataset:"
)

print(
    f"Test samples: {len(X_test)}"
)


# --------------------------------------------------
# 10. Completion
# --------------------------------------------------

print(
    "\n==================================="
)

print(
    "Federated CKD Training Completed."
)

print(
    "Global model evaluation completed."
)

print(
    "==================================="
)
# --------------------------------------------------
# 11. XAI / SHAP Explanation
# --------------------------------------------------

from xai.shap_explainer import (
    display_explanation
)


print(
    "\nStarting XAI / SHAP explanation..."
)


# Use the first test patient as an example
patient_index = 0

X_patient = X_test.iloc[
    [patient_index]
]


feature_names = list(
    X_test.columns
)


xai_result = display_explanation(
    global_model,
    X_train,
    X_patient,
    feature_names,
    target_encoder
)


print(
    "\nXAI explanation generated successfully."
)
# ===================================
# SAVE GLOBAL MODEL
# ===================================

import os
import joblib

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(global_model, os.path.join(MODEL_DIR, "global_ckd_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.pkl"))

print("\n===================================")
print("       MODEL SAVING")
print("===================================")

print("Global model saved successfully.")
print("Scaler saved successfully.")
print("Label encoders saved successfully.")
print("Target encoder saved successfully.")
# ===================================
# SAVE SHAP BACKGROUND DATA
# ===================================

joblib.dump(
    X_train,
    os.path.join(MODEL_DIR, "X_train_background.pkl")
)

print("SHAP background training data saved successfully.")