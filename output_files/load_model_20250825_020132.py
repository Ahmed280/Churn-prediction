#!/usr/bin/env python3
"""
Model Loading Script for Churn Prediction Model
Generated: 2025-08-25 02:01:32
"""

import joblib
import json
import pandas as pd
import numpy as np

def load_churn_model():
    """Load the trained churn prediction model and its metadata."""
    
    # Load the model
    model = joblib.load('models\final_churn_model_20250825_020132.joblib')
    
    # Load feature columns
    with open('output_files\feature_columns_20250825_020132.json', 'r') as f:
        feature_columns = json.load(f)
    
    # Load model report
    with open('output_files\model_report_20250825_020132.json', 'r') as f:
        model_report = json.load(f)
    
    print(f"✅ Loaded model: {model_report['model_info']['type']}")
    print(f"📊 Test F1 Score: {model_report['performance']['f1']:.4f}")
    print(f"📊 Test Accuracy: {model_report['performance']['accuracy']:.4f}")
    print(f"🔢 Features: {len(feature_columns)}")
    
    return model, feature_columns, model_report

def predict_churn(model, feature_columns, user_data):
    """Make churn predictions on new user data."""
    
    # Ensure user_data has the required columns
    if isinstance(user_data, dict):
        user_data = pd.DataFrame([user_data])
    
    # Reorder columns to match training data
    user_data = user_data.reindex(columns=feature_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(user_data)
    probability = model.predict_proba(user_data)[:, 1] if hasattr(model, 'predict_proba') else None
    
    return prediction, probability

if __name__ == "__main__":
    # Example usage
    model, features, report = load_churn_model()
    print(f"\n🎯 Model ready for predictions!")
    print(f"Example: prediction, probability = predict_churn(model, features, user_data_dict)")
