
from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, flash
)
import joblib
import os
import numpy as np
import pandas as pd
import shap
import sqlite3
import re

from datetime import timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash


# ===================================
# FLASK CONFIGURATION
# ===================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-secret-key-before-deployment"
)

app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "users.db"
)


# ===================================
# DATABASE CONNECTION
# ===================================

def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth_db():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            if request.path == "/predict":
                return jsonify({
                    "error": "Please sign in to use CKD prediction."
                }), 401

            return redirect(url_for("signin"))

        return view(*args, **kwargs)

    return wrapped_view


initialize_auth_db()


# ===================================
# LOAD SAVED MODEL AND PREPROCESSING
# ===================================

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models"
)

global_model = joblib.load(
    os.path.join(MODEL_DIR, "global_ckd_model.pkl")
)

if set(np.asarray(global_model.classes_).tolist()) != {0, 1}:
    raise ValueError(
        f"Unexpected model classes {global_model.classes_}; "
        "expected encoded classes [0, 1]."
    )

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

label_encoders = joblib.load(
    os.path.join(MODEL_DIR, "label_encoders.pkl")
)

target_encoder = joblib.load(
    os.path.join(MODEL_DIR, "target_encoder.pkl")
)

PREPROCESS_PATH = os.path.join(
    MODEL_DIR,
    "preprocessing_artifacts.pkl"
)

if not os.path.exists(PREPROCESS_PATH):
    raise FileNotFoundError(
        "Missing preprocessing_artifacts.pkl. Run python main.py "
        "to retrain and save the fitted preprocessing objects "
        "before starting the web app."
    )

preprocessing_artifacts = joblib.load(PREPROCESS_PATH)

numerical_imputer = preprocessing_artifacts["numerical_imputer"]
categorical_imputer = preprocessing_artifacts["categorical_imputer"]


# SHAP background data
X_train_background = joblib.load(
    os.path.join(MODEL_DIR, "X_train_background.pkl")
)

print("Global CKD model loaded successfully.")
print("Preprocessing objects loaded successfully.")
print("SHAP background data loaded successfully.")


# ===================================
# FEATURE ORDER
# ===================================

FEATURE_NAMES = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc",
    "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet",
    "pe", "ane"
]


# ===================================
# NUMERICAL FEATURES
# ===================================

NUMERICAL_COLUMNS = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc"
]


# ===================================
# CATEGORICAL FEATURES
# ===================================

CATEGORICAL_COLUMNS = [
    "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet",
    "pe", "ane"
]


# ===================================
# PREPROCESS PATIENT DATA
# ===================================

def preprocess_patient(patient_data):
    """Apply the same fitted imputers, encoders and scaler used in training."""

    df = pd.DataFrame(
        [patient_data],
        columns=FEATURE_NAMES
    )

    for column in NUMERICAL_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Treat blank strings as missing.
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].apply(
            lambda value: (
                np.nan
                if value is None or str(value).strip() == ""
                else str(value).strip()
            )
        )

    numerical_array = numerical_imputer.transform(
        df[NUMERICAL_COLUMNS]
    )

    numerical_data = pd.DataFrame(
        numerical_array,
        columns=NUMERICAL_COLUMNS,
        index=df.index
    )

    numerical_scaled = pd.DataFrame(
        scaler.transform(numerical_data),
        columns=NUMERICAL_COLUMNS,
        index=df.index
    )

    categorical_array = categorical_imputer.transform(
        df[CATEGORICAL_COLUMNS]
    )

    categorical_data = pd.DataFrame(
        categorical_array,
        columns=CATEGORICAL_COLUMNS,
        index=df.index
    )

    categorical_encoded = pd.DataFrame(index=df.index)

    for column in CATEGORICAL_COLUMNS:
        encoder = label_encoders[column]
        value = str(categorical_data.loc[0, column]).strip()

        if value not in encoder.classes_:
            raise ValueError(
                f"Invalid value '{value}' for {column}. Allowed values: "
                + ", ".join(map(str, encoder.classes_))
            )

        categorical_encoded[column] = encoder.transform([value])

    processed = pd.concat(
        [numerical_scaled, categorical_encoded],
        axis=1
    )

    return processed[FEATURE_NAMES]


# ===================================
# SHAP EXPLANATION
# ===================================

def generate_shap_explanation(patient_data):

    # Create SHAP explainer
    explainer = shap.LinearExplainer(
        global_model,
        X_train_background
    )

    # Generate SHAP values
    shap_values = explainer.shap_values(
        patient_data
    )

    # Convert to numpy
    shap_values = np.asarray(shap_values)

    # Handle different SHAP output shapes
    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    # SHAP from the model represents class 1 = notCKD.
    # Convert to CKD-oriented SHAP.
    ckd_shap_values = -shap_values

    explanations = []

    for index, feature in enumerate(FEATURE_NAMES):

        shap_value = float(
            ckd_shap_values[index]
        )

        feature_value = patient_data.iloc[0, index]

        if shap_value > 0:
            direction = "Supports CKD"
        else:
            direction = "Supports not CKD"

        explanations.append({
            "feature": feature,
            "value": round(float(feature_value), 4),
            "shap_value": round(shap_value, 4),
            "direction": direction
        })

    # Sort by absolute contribution
    explanations.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    # Return top 8 features
    return explanations[:8]


# ===================================
# HOME PAGE
# ===================================

@app.route("/")
@login_required
def home():
    return render_template(
        "index.html",
        user_name=session.get("user_name")
    )


# ===================================
# SIGN IN
# ===================================

@app.route("/signin", methods=["GET", "POST"])
def signin():

    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        remember = request.form.get("remember") == "on"

        with get_db_connection() as connection:
            user = connection.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            ).fetchone()

        if user and check_password_hash(
            user["password_hash"],
            password
        ):

            session.clear()

            session["user_id"] = user["id"]
            session["user_name"] = user["full_name"]

            # Remember Me: keep session for up to 30 days.
            session.permanent = remember

            return redirect(url_for("home"))

        flash(
            "Invalid email or password. Please try again.",
            "error"
        )

    return render_template("signin.html")


# ===================================
# SIGN UP
# ===================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":

        full_name = request.form.get(
            "full_name", ""
        ).strip()

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if len(full_name) < 2:
            flash(
                "Please enter your full name.",
                "error"
            )

        elif not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            email
        ):
            flash(
                "Please enter a valid email address.",
                "error"
            )

        elif len(password) < 8:
            flash(
                "Password must contain at least 8 characters.",
                "error"
            )

        elif password != confirm_password:
            flash(
                "Passwords do not match.",
                "error"
            )

        else:
            try:
                with get_db_connection() as connection:

                    cursor = connection.execute(
                        """
                        INSERT INTO users
                        (full_name, email, password_hash)
                        VALUES (?, ?, ?)
                        """,
                        (
                            full_name,
                            email,
                            generate_password_hash(password)
                        )
                    )

                    user_id = cursor.lastrowid

                session.clear()

                session["user_id"] = user_id
                session["user_name"] = full_name
                session.permanent = True

                return redirect(url_for("home"))

            except sqlite3.IntegrityError:
                flash(
                    "An account with this email already exists. "
                    "Please sign in.",
                    "error"
                )

    return render_template("signup.html")


# ===================================
# LOGOUT
# ===================================

@app.route("/logout", methods=["POST"])
def logout():

    session.clear()

    return redirect(url_for("signin"))


# ===================================
# PREDICTION API
# ===================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No patient data provided."
            }), 400

        # Check missing features
        missing_features = [
            feature
            for feature in FEATURE_NAMES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing features.",
                "missing_features": missing_features
            }), 400

        # Preprocess patient
        processed_data = preprocess_patient(data)

        X_patient = processed_data.to_numpy()

        # Prediction
        prediction = global_model.predict(
            X_patient
        )[0]

        probabilities = global_model.predict_proba(
            X_patient
        )[0]

        # Convert encoded class back to original class
        predicted_class = target_encoder.inverse_transform(
            [prediction]
        )[0]

        # Get CKD probability
        ckd_index = list(
            global_model.classes_
        ).index(0)

        not_ckd_index = list(
            global_model.classes_
        ).index(1)

        ckd_probability = probabilities[ckd_index]

        not_ckd_probability = probabilities[not_ckd_index]

        # SHAP explanation
        shap_explanation = generate_shap_explanation(
            processed_data
        )

        # Response
        return jsonify({
            "prediction": predicted_class,
            "ckd_probability": float(ckd_probability),
            "not_ckd_probability": float(not_ckd_probability),
            "shap_explanation": shap_explanation
        })

    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ===================================
# RUN SERVER
# ===================================

if __name__ == "__main__":

    print("\n===================================")
    print("       CKD FLASK WEB APPLICATION")
    print("===================================")

    print("Open in browser:")
    print("http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
