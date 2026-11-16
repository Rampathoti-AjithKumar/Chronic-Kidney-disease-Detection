import numpy as np
import os
import joblib
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "chronic_kidney_disease.arff")
CLEANED_DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "cleaned_ckd.arff")
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")

RANDOM_STATE = 42

def load_dataset():

    print("\n===================================")
    print("       DATASET INFORMATION")
    print("===================================")

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()

    cleaned_lines = []

    for line in lines:

        stripped = line.strip()

        if (
            not stripped
            or stripped.startswith("@")
        ):
            cleaned_lines.append(line)
            continue

        parts = stripped.split(",")

        parts = [
            part.strip()
            for part in parts
        ]

        if len(parts) == 25:

            cleaned_lines.append(
                ",".join(parts) + "\n"
            )

        elif (
            len(parts) == 26
            and parts[-1] == ""
        ):

            parts.pop()

            cleaned_lines.append(
                ",".join(parts) + "\n"
            )

        elif (
            len(parts) == 26
            and parts.count("") == 1
        ):

            empty_index = parts.index("")

            parts.pop(empty_index)

            cleaned_lines.append(
                ",".join(parts) + "\n"
            )

        else:

            print(
                f"Warning: unexpected row with "
                f"{len(parts)} fields:"
            )

            print(stripped)

            continue

    with open(
        CLEANED_DATASET_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.writelines(cleaned_lines)

    data, meta = arff.loadarff(
        CLEANED_DATASET_PATH
    )

    df = pd.DataFrame(
        data,
        columns=meta.names()
    )

    for column in df.columns:

        if df[column].dtype == object:

            df[column] = df[column].apply(
                lambda value:
                value.decode("utf-8")
                if isinstance(value, bytes)
                else value
            )

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print(
        f"\nSUCCESS: All {len(df)} CKD records loaded."
    )

    return df


def clean_dataset(df):

    df = df.copy()

    df = df.replace(
        "?",
        np.nan
    )

    for column in df.select_dtypes(
        include=["object"]
    ).columns:

        df[column] = df[column].apply(
            lambda value:
            value.strip()
            if isinstance(value, str)
            else value
        )

    return df


def prepare_raw_features(df):

    X = df.drop(
        columns=["class"]
    ).copy()

    y = df["class"].copy()

    numerical_columns = [
        "age",
        "bp",
        "sg",
        "al",
        "su",
        "bgr",
        "bu",
        "sc",
        "sod",
        "pot",
        "hemo",
        "pcv",
        "wbcc",
        "rbcc"
    ]

    numerical_columns = [
        column
        for column in numerical_columns
        if column in X.columns
    ]

    for column in numerical_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    categorical_columns = [
        column
        for column in X.columns
        if column not in numerical_columns
    ]

    return (
        X,
        y,
        numerical_columns,
        categorical_columns
    )


def fit_preprocessors(
    X_train,
    numerical_columns,
    categorical_columns
):

    numerical_imputer = SimpleImputer(
        strategy="median"
    )

    numerical_training_array = (
        numerical_imputer.fit_transform(
            X_train[numerical_columns]
        )
    )

    numerical_training_data = pd.DataFrame(
        numerical_training_array,
        columns=numerical_columns,
        index=X_train.index
    )

    scaler = StandardScaler()

    scaler.fit(
        numerical_training_data
    )

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    categorical_training_array = (
        categorical_imputer.fit_transform(
            X_train[categorical_columns]
        )
    )

    categorical_training_data = pd.DataFrame(
        categorical_training_array,
        columns=categorical_columns,
        index=X_train.index
    )

    label_encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        encoder.fit(
            categorical_training_data[column].astype(str)
        )

        label_encoders[column] = encoder

    return (
        numerical_imputer,
        categorical_imputer,
        scaler,
        label_encoders
    )


def transform_features(
    X,
    numerical_columns,
    categorical_columns,
    numerical_imputer,
    categorical_imputer,
    scaler,
    label_encoders
):

    X = X.copy()

    numerical_array = (
        numerical_imputer.transform(
            X[numerical_columns]
        )
    )

    numerical_data = pd.DataFrame(
        numerical_array,
        columns=numerical_columns,
        index=X.index
    )

    numerical_scaled_array = scaler.transform(
        numerical_data
    )

    numerical_scaled = pd.DataFrame(
        numerical_scaled_array,
        columns=numerical_columns,
        index=X.index
    )

    categorical_array = (
        categorical_imputer.transform(
            X[categorical_columns]
        )
    )

    categorical_data = pd.DataFrame(
        categorical_array,
        columns=categorical_columns,
        index=X.index
    )

    categorical_encoded = pd.DataFrame(
        index=X.index
    )

    for column in categorical_columns:

        encoder = label_encoders[column]

        categorical_encoded[column] = (
            encoder.transform(
                categorical_data[column].astype(str)
            )
        )

    processed_data = pd.concat(
        [
            numerical_scaled,
            categorical_encoded
        ],
        axis=1
    )

    processed_data = processed_data[
        numerical_columns +
        categorical_columns
    ]

    return processed_data


def prepare_data():

    df = load_dataset()

    df = clean_dataset(df)

    (
        X,
        y,
        numerical_columns,
        categorical_columns
    ) = prepare_raw_features(df)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    (
        numerical_imputer,
        categorical_imputer,
        scaler,
        label_encoders
    ) = fit_preprocessors(
        X_train,
        numerical_columns,
        categorical_columns
    )

    X_train_processed = transform_features(
        X_train,
        numerical_columns,
        categorical_columns,
        numerical_imputer,
        categorical_imputer,
        scaler,
        label_encoders
    )

    X_test_processed = transform_features(
        X_test,
        numerical_columns,
        categorical_columns,
        numerical_imputer,
        categorical_imputer,
        scaler,
        label_encoders
    )

    target_encoder = LabelEncoder()

    y_train_encoded = (
        target_encoder.fit_transform(
            y_train
        )
    )

    y_test_encoded = (
        target_encoder.transform(
            y_test
        )
    )

    # Persist the exact fitted preprocessing objects used for training.
    # The web application must reuse these objects instead of approximating imputation.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.pkl"))
    joblib.dump({
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "numerical_imputer": numerical_imputer,
        "categorical_imputer": categorical_imputer,
        "feature_names": list(X_train_processed.columns),
    }, os.path.join(MODEL_DIR, "preprocessing_artifacts.pkl"))

    print("\nSaved fitted preprocessing artifacts for inference.")

    print("\n===================================")
    print("       PREPROCESSING RESULT")
    print("===================================")

    print("\nTraining Data Shape:")
    print(
        X_train_processed.shape
    )

    print("\nTesting Data Shape:")
    print(
        X_test_processed.shape
    )

    print("\nTarget Classes:")
    print(
        target_encoder.classes_
    )

    print("\nTarget Mapping:")

    for index, class_name in enumerate(
        target_encoder.classes_
    ):

        print(
            f"  {index} -> {class_name}"
        )

    print("\nFeature Names:")

    for feature in X_train_processed.columns:

        print(
            f"  {feature}"
        )

    print(
        "\nPreprocessing completed successfully."
    )

    return (
        X_train_processed,
        X_test_processed,

        pd.Series(
            y_train_encoded,
            index=X_train_processed.index
        ),

        pd.Series(
            y_test_encoded,
            index=X_test_processed.index
        ),

        scaler,
        label_encoders,
        target_encoder
    )


if __name__ == "__main__":

    prepare_data()