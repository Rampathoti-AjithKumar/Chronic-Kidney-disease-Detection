# CKD Federated Learning + XAI — Updated Files

## What was updated
- `preprocessing/preprocess.py`: saves the fitted numerical/categorical imputers used during training; dataset paths are anchored to the project directory.
- `web/app.py`: reuses the exact saved imputers and scaler at inference; rejects unknown categorical values instead of silently converting them to the first known category; returns full-precision probabilities.
- `models/ckd_model.py`: stronger SGD regularization and more training iterations. This changes the model and requires retraining.
- `clients/client_data.py`: deterministic client partitioning for repeatable experiments.
- `main.py`: prints a diagnostic summary of test-set probability extremes.

## Important
The changes do not guarantee that all probabilities become intermediate percentages. A model may legitimately return values very close to 0 or 1, and probabilities can be poorly calibrated. Do not manually replace or clamp them. Evaluate calibration separately using a calibration split or cross-validation that does not touch the final test set.

## Run from the project root
1. Back up your current project and saved models.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Retrain and regenerate all saved artifacts: `python main.py`
4. Start the web app: `python web/app.py`
5. Open `http://127.0.0.1:5000`

Run `main.py` before starting the web app. It regenerates `saved_models/global_ckd_model.pkl`, the fitted preprocessing artifacts, scaler, encoders, and SHAP background data.

The test set remains held out from model fitting in the training workflow. The current federated setup is a simulation using partitions of one dataset, not a real multi-hospital deployment.


## Clinical portal UI
The web interface has been updated to follow the supplied hospital clinical portal references. Sign-in and registration use the existing Flask session and SQLite account routes. The patient assessment is organized into four tabs and retains the existing /predict API and SHAP response format.
