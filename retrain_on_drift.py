"""
Automated Retraining Entrypoint Triggered by Drift.

Loads latest events, re-creates features/labels with leak-safety, evaluates
performance & data drift vs. the packaged baseline, and—if thresholds are
exceeded—tunes and writes a new production package.

This module is **orchestration-only**; modeling primitives live in `split.py`,
`eval.py`, `monitor.py`, and `MusicStreamingEventProcessor.py`.

Author: Ahmed Alghaith
Date: August 2025
"""

import joblib
import json
import pandas as pd
from datetime import datetime
from monitor import ChurnModelMonitoringSystem
from split import prepare_training_data, get_model_configurations
from eval import train_and_evaluate_models, robust_hyperparameter_tuning
from MusicStreamingEventProcessor import MusicStreamingEventProcessor

def load_data(path="customer_churn.json"):
    df = pd.read_json(path, lines=True)
    return df

def retrain_if_drift(event_log_path="customer_churn.json",
                     model_pkg_path="model_package.pkl",
                     drift_threshold=0.20):
    """
    Retrain the best model if drift is detected.

    Args:
        event_log_path: Path to raw JSON lines event logs.
        model_pkg_path: Path to an existing packaged model (Joblib).
        drift_threshold: Ratio of features allowed to drift before triggering.

    Returns:
        dict | None: Newly saved package metadata if retrained, else None.
    """
    # 1. Load raw events
    events_df = load_data(event_log_path)

    # 2. Process features & labels
    processor = MusicStreamingEventProcessor()
    cleaned = processor.clean_events(events_df)
    features = processor.engineer_user_features()
    labels = processor.identify_churn_users()
    features["churn"] = features["userId"].map(labels).fillna(0).astype(int)

    # 3. Temporal split
    train_df, val_df, test_df = prepare_training_data(features)
    X_train, y_train = train_df.drop(["userId","churn"], axis=1), train_df["churn"]
    X_val, y_val     = val_df.drop(["userId","churn"], axis=1), val_df["churn"]

    # 4. Load existing model & baseline
    pkg = joblib.load(model_pkg_path)
    model = pkg["model"]
    cols  = pkg["feature_columns"]
    baseline_metrics = pkg.get("performance", {})

    # 5. Monitor drift
    monitor = ChurnModelMonitoringSystem(model, baseline_metrics, cols)
    report = monitor.generate_monitoring_report(X_new=X_val, y_new=y_val, X_orig=X_train)

    # 6. Check retraining condition
    if report["recommendation"]["retrain"]:
        print("⚠️ Data/proformance drift detected – retraining model...")
        # 7. Configure & retrain best model only
        models_config = get_model_configurations((X_train, y_train))
        # assume best_model_name stored in pkg metadata
        best_name = pkg["model_name"]
        single_cfg = {best_name: models_config[best_name]}

        # 8. Hyperparameter tuning on best model
        results, new_model = robust_hyperparameter_tuning(
            model_configs=single_cfg,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            use_optuna=True
        )

        # 9. Evaluate on validation
        new_metrics = new_model and train_and_evaluate_models(
            {best_name: models_config[best_name]}, X_val, y_val
        )[1].set_index("Model").loc[best_name].to_dict()

        # 10. Save new package
        new_pkg = {
            "model": new_model,
            "feature_columns": cols,
            "model_name": best_name,
            "performance": new_metrics,
            "timestamp": datetime.now().isoformat()
        }
        out_path = f"{best_name.replace(' ', '_')}_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(new_pkg, out_path)
        print(f"✅ Retrained model saved to {out_path}")
        return new_pkg
    else:
        print("✅ No significant drift detected – no retraining needed.")
        return None

if __name__ == "__main__":
    retrain_if_drift()
