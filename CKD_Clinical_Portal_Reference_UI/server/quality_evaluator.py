from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ---------------------------------------------------------
# 1. Calculate performance of a local model
# ---------------------------------------------------------

def calculate_performance(model, X_test, y_test):

    # Make predictions
    predictions = model.predict(X_test)

    # Get prediction probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# ---------------------------------------------------------
# 2. Get performance score
# ---------------------------------------------------------

def get_performance_score(model, X_test, y_test):

    metrics = calculate_performance(
        model,
        X_test,
        y_test
    )

    # F1-score is used as the performance component
    performance_score = metrics["f1_score"]

    return performance_score, metrics