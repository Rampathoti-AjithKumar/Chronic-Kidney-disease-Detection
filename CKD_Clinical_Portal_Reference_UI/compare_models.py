
import os
import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from preprocessing.preprocess import prepare_data


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_DIR = "saved_models"

LOGISTIC_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "global_ckd_model.pkl"
)

SVM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "svm_ckd_model.pkl"
)

# Target mapping:
# 0 = CKD
# 1 = notCKD
CKD_CLASS = 0


# ==================================================
# MODEL EVALUATION
# ==================================================

def evaluate_model(name, model, X_test, y_test):

    predictions = np.asarray(
        model.predict(X_test)
    )

    y_test = np.asarray(y_test)

    # Probability of CKD (class 0)
    probabilities = model.predict_proba(X_test)

    ckd_index = list(model.classes_).index(
        CKD_CLASS
    )

    ckd_probabilities = probabilities[
        :, ckd_index
    ]

    # Treat CKD as the positive class
    actual_ckd = (
        y_test == CKD_CLASS
    ).astype(int)

    predicted_ckd = (
        predictions == CKD_CLASS
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        actual_ckd,
        predicted_ckd,
        zero_division=0
    )

    recall = recall_score(
        actual_ckd,
        predicted_ckd,
        zero_division=0
    )

    f1 = f1_score(
        actual_ckd,
        predicted_ckd,
        zero_division=0
    )

    auc = roc_auc_score(
        actual_ckd,
        ckd_probabilities
    )

    print("\n" + "=" * 45)
    print(f"{name} RESULTS")
    print("=" * 45)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix")
    print("Rows = Actual, Columns = Predicted")
    print("Order: CKD, notCKD")

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print(matrix)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }


# ==================================================
# MAIN FUNCTION
# ==================================================

def main():

    print("\nLoading and preprocessing CKD dataset...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        label_encoders,
        target_encoder
    ) = prepare_data()

    # Convert features and labels to NumPy arrays
    X_train = X_train.to_numpy(
        dtype=float
    )

    X_test = X_test.to_numpy(
        dtype=float
    )

    y_train = np.asarray(
        y_train,
        dtype=int
    )

    y_test = np.asarray(
        y_test,
        dtype=int
    )

    # ----------------------------------------------
    # 1. LOAD EXISTING LOGISTIC MODEL
    # ----------------------------------------------

    if not os.path.exists(
        LOGISTIC_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"\nExisting Logistic model not found:\n"
            f"{LOGISTIC_MODEL_PATH}\n"
            "Run your existing main.py successfully first."
        )

    logistic_model = joblib.load(
        LOGISTIC_MODEL_PATH
    )

    print(
        "\nExisting Logistic model loaded successfully."
    )

    print(
        "The original Logistic model will not be overwritten."
    )

    logistic_results = evaluate_model(
        "EXISTING FEDERATED LOGISTIC MODEL",
        logistic_model,
        X_test,
        y_test
    )

    # ----------------------------------------------
    # 2. CREATE A NEW SVM MODEL
    # ----------------------------------------------

    print("\nCreating a separate SVM model...")

    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    )

    # ----------------------------------------------
    # 3. FIVE-FOLD CROSS-VALIDATION
    # ----------------------------------------------

    print(
        "\nRunning 5-fold SVM cross-validation..."
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        svm_model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print("\nCross-Validation Results")

    for index, score in enumerate(
        scores,
        start=1
    ):
        print(
            f"Fold {index} Accuracy: "
            f"{score * 100:.2f}%"
        )

    mean_accuracy = scores.mean() * 100

    std_accuracy = scores.std() * 100

    print(
        f"\nMean CV Accuracy: {mean_accuracy:.2f}%"
    )

    print(
        f"Standard Deviation: {std_accuracy:.2f}%"
    )

    # ----------------------------------------------
    # 4. TRAIN FINAL SVM ON TRAINING DATA
    # ----------------------------------------------

    print(
        "\nTraining SVM on the complete training set..."
    )

    svm_model.fit(
        X_train,
        y_train
    )

    # ----------------------------------------------
    # 5. EVALUATE SVM ON HELD-OUT TEST DATA
    # ----------------------------------------------

    svm_results = evaluate_model(
        "SVM (RBF KERNEL)",
        svm_model,
        X_test,
        y_test
    )

    # ----------------------------------------------
    # 6. SAVE SVM SEPARATELY
    # ----------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        svm_model,
        SVM_MODEL_PATH
    )

    print(
        f"\nSVM model saved at:\n{SVM_MODEL_PATH}"
    )

    print(
        "\nOriginal Logistic model remains unchanged."
    )

    # ----------------------------------------------
    # 7. COMPARE BOTH MODELS
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST-SET MODEL COMPARISON")
    print("=" * 60)

    print(
        f"{'Metric':<15}"
        f"{'Logistic':>15}"
        f"{'SVM':>15}"
    )

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    for metric in metrics:

        logistic_value = logistic_results[
            metric
        ]

        svm_value = svm_results[
            metric
        ]

        print(
            f"{metric:<15}"
            f"{logistic_value:>15.4f}"
            f"{svm_value:>15.4f}"
        )

    # ----------------------------------------------
    # 8. SUMMARY
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"SVM mean cross-validation accuracy: "
        f"{mean_accuracy:.2f}%"
    )

    print(
        f"Logistic test accuracy: "
        f"{logistic_results['accuracy'] * 100:.2f}%"
    )

    print(
        f"SVM test accuracy: "
        f"{svm_results['accuracy'] * 100:.2f}%"
    )

    if (
        svm_results["f1_score"]
        > logistic_results["f1_score"]
    ):
        print(
            "\nSVM achieved a higher CKD F1-score "
            "on this test set."
        )

    elif (
        svm_results["f1_score"]
        < logistic_results["f1_score"]
    ):
        print(
            "\nLogistic Regression achieved a higher "
            "CKD F1-score on this test set."
        )

    else:
        print(
            "\nBoth models achieved the same CKD F1-score."
        )

    print(
        "\nComparison completed successfully."
    )


# ==================================================
# PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
