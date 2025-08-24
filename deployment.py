"""
Production Packaging & Single-Record Prediction

Bundles model and metadata into a single joblib package for reproducible
inference. Includes a convenience function for single-user predictions.

Author: Ahmed Alghaith
Date: August 2025
"""

import joblib
import json
from datetime import datetime

def deploy_model(model, model_name: str, feature_columns: list, performance_metrics: dict, save_path: str = "."):
    """
    Persist model + metadata for production use.

    Args:
        model: Trained estimator.
        model_name: Stable identifier (used in filename).
        feature_columns: Ordered list used by the model.
        performance_metrics: Dict of evaluation metrics.
        save_path: Directory where the package is written.

    Returns:
        None. Writes `<model_name>_production.pkl` to disk.
    """
    print("💾 Preparing model for production deployment...")

    # Build metadata
    model_metadata = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "training_timestamp": datetime.now().isoformat(),
        "performance": performance_metrics,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns)
    }

    # Save model + metadata together
    pkg = {
        "model": model,
        "metadata": model_metadata
    }
    filename = f"{save_path}/{model_name.replace(' ', '_')}_production.pkl"
    joblib.dump(pkg, filename)
    print(f"✅ Saved model+metadata to {filename}")

def production_predict_churn(user_features: dict, pkg_path: str):
    """
    Load package and score a **single** user feature dict.

    Args:
        user_features: Mapping feature_name → value.
        pkg_path: Path to `.pkl` created by `deploy_model()`.

    Returns:
        Dict with `churn_prediction`, `churn_probability`, `risk_level`,
        `model_name`, `prediction_timestamp`, `feature_count`.
    """
    # Load package
    pkg = joblib.load(pkg_path)
    model = pkg["model"]
    metadata = pkg["metadata"]
    feat_cols = metadata["feature_columns"]

    # Build DataFrame
    import pandas as pd
    df = pd.DataFrame([user_features])
    # Ensure all features exist
    for c in feat_cols:
        if c not in df:
            df[c] = 0
    X = df[feat_cols]

    # Predict
    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0, 1] if model.predict_proba(X).shape[1] > 1 else 0.5
    else:
        prob = float(pred)

    # Assign risk
    if prob < 0.3:
        risk = "Low"
    elif prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_prediction": int(pred),
        "churn_probability": float(prob),
        "risk_level": risk,
        "model_name": metadata["model_name"],
        "prediction_timestamp": datetime.now().isoformat(),
        "feature_count": metadata["feature_count"]
    }
