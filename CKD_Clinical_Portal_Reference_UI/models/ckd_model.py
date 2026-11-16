import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# In the dataset:
# 0 = CKD
# 1 = notCKD
CKD_CLASS = 0


def _to_numpy(X):
    """
    Convert pandas DataFrame/Series or other array-like input
    into NumPy arrays.

    This also prevents sklearn feature-name warnings.
    """
    if hasattr(X, "to_numpy"):
        return X.to_numpy()

    return np.asarray(X)


def create_model():
    """
    Create the local CKD classification model.
    """

    model = SGDClassifier(
        loss="log_loss",
        max_iter=2000,
        tol=1e-4,
        alpha=0.001,
        random_state=42,
        warm_start=True
    )

    return model


def train_model(model, X_train, y_train):
    """
    Train the model using local hospital/client data.
    """

    X_train = _to_numpy(X_train)
    y_train = _to_numpy(y_train)

    model.fit(X_train, y_train)

    return model


def predict(model, X):
    """
    Predict CKD / notCKD class.
    """

    X = _to_numpy(X)

    return model.predict(X)


def predict_probability(model, X):
    """
    Return CKD probability.

    Dataset mapping:
        0 = CKD
        1 = notCKD
    """

    X = _to_numpy(X)

    probabilities = model.predict_proba(X)

    # Find the position of CKD class in model.classes_
    ckd_index = list(model.classes_).index(CKD_CLASS)

    return probabilities[:, ckd_index]


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model with CKD treated as the positive class.
    """

    X_test = _to_numpy(X_test)
    y_test = _to_numpy(y_test)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    # Find CKD probability
    ckd_index = list(model.classes_).index(CKD_CLASS)
    ckd_probabilities = probabilities[:, ckd_index]

    # Convert to CKD-positive binary representation
    # CKD     -> 1
    # notCKD  -> 0
    actual_ckd = (y_test == CKD_CLASS).astype(int)
    predicted_ckd = (predictions == CKD_CLASS).astype(int)

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

    # ROC-AUC must use probability of the positive class,
    # which is CKD in our project.
    auc = roc_auc_score(
        actual_ckd,
        ckd_probabilities
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }


def get_model_parameters(model):
    """
    Extract model parameters for federated aggregation.
    """

    return [
        model.coef_.copy(),
        model.intercept_.copy()
    ]


def set_model_parameters(model, parameters):
    """
    Load aggregated global parameters into a model.
    """

    model.coef_ = np.array(parameters[0])
    model.intercept_ = np.array(parameters[1])

    # Dataset class mapping:
    # 0 = CKD
    # 1 = notCKD
    model.classes_ = np.array([0, 1])

    return model


if __name__ == "__main__":
    print("CKD model module loaded successfully.")
    print("Class mapping: 0 = CKD, 1 = notCKD")