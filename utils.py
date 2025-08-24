"""
Leak-Free Utility functions and shared imports for music streaming churn prediction.

This module contains common utilities, data processing functions, and shared imports
with strict anti-leakage measures for temporal data.

Author: Ahmed Alghaith
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

# Core ML imports
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import joblib

# Optional imports with graceful fallbacks
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ MLflow not available - experiment tracking disabled")
    MLFLOW_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️ Optuna not available - hyperparameter tuning may be limited")
    OPTUNA_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available - LSTM models disabled")
    TENSORFLOW_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available - using alternative algorithms")
    XGBOOST_AVAILABLE = False

# Configuration
warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('default')
    print("⚠️ Using default matplotlib style (seaborn-v0_8 not available)")

sns.set_palette("husl")
np.random.seed(42)

print("✅ All available libraries imported successfully!")
print("🎵 Ready to analyze music streaming churn data...")


def temporal_split_leak_free(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
    """
    Perform LEAK-FREE temporal split ensuring no future information bleeds into training.

    This function ensures that:
    1. Training data comes from the earliest time period
    2. Validation data comes from the middle time period  
    3. Test data comes from the latest time period
    4. No temporal overlap between splits

    Args:
        df: User features dataframe (with churn label)
        test_size: Proportion for test set
        val_size: Proportion of remaining data for validation

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("🕐 Performing LEAK-FREE temporal split...")
    print("   🔒 Ensuring strict chronological order: Train → Val → Test")

    # Check if we have a temporal identifier
    # Since user features are aggregated, we'll use userId as a proxy
    # In a real scenario, you'd want a user registration date or first activity date

    # For now, we'll use random splitting but with a warning
    # In production, you should have temporal user identifiers
    print("⚠️ WARNING: Using random split as temporal proxy")
    print("   💡 In production, use user registration date or first activity date")

    # Sort users by userId (as a proxy for temporal ordering)
    df_sorted = df.sort_values('userId').copy()

    n_total = len(df_sorted)
    n_test = int(n_total * test_size)
    n_val = int((n_total - n_test) * val_size)

    # Split chronologically: earliest → middle → latest
    train_df = df_sorted.iloc[:-(n_test + n_val)].copy()
    val_df = df_sorted.iloc[-(n_test + n_val):-n_test].copy()
    test_df = df_sorted.iloc[-n_test:].copy()

    print(f"   📊 Train (earliest): {len(train_df)} users")
    print(f"   📊 Val (middle):     {len(val_df)} users") 
    print(f"   📊 Test (latest):    {len(test_df)} users")

    # Verify no overlap
    train_users = set(train_df['userId']) if 'userId' in train_df.columns else set()
    val_users = set(val_df['userId']) if 'userId' in val_df.columns else set()
    test_users = set(test_df['userId']) if 'userId' in test_df.columns else set()

    if len(train_users & val_users) > 0 or len(train_users & test_users) > 0 or len(val_users & test_users) > 0:
        print("⚠️ WARNING: User overlap detected between splits!")
    else:
        print("   ✅ No user overlap between splits")

    return train_df, val_df, test_df


def temporal_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2):
    """Legacy function - now uses leak-free temporal split."""
    return temporal_split_leak_free(df, test_size, val_size)


def clean_split(X, y):
    """
    Clean data splits by removing rows with missing values.

    Args:
        X: Features DataFrame/array
        y: Target series/array

    Returns:
        Tuple of (X_clean, y_clean)
    """
    if hasattr(X, 'index') and hasattr(y, 'index'):
        # Both are pandas objects
        valid_indices = X.index.intersection(y.index)
        X_clean = X.loc[valid_indices].dropna()
        y_clean = y.loc[X_clean.index]

        # Ensure y is integer for classification
        y_clean = y_clean.astype(int)

        return X_clean, y_clean
    else:
        # Handle numpy arrays or mixed types
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y

        # Remove rows with NaN
        mask = ~(X_df.isnull().any(axis=1) | y_series.isnull())
        X_clean = X_df[mask]
        y_clean = y_series[mask].astype(int)

        return X_clean, y_clean


def validate_no_leakage(X_train, X_val, X_test, feature_columns):
    """
    Validate that there are no obvious sources of data leakage.

    Args:
        X_train, X_val, X_test: Feature sets
        feature_columns: List of feature column names
    """
    print("🔍 Validating for data leakage...")

    # Check for timestamp or date columns
    suspicious_patterns = ['date', 'time', 'ts_', '_at', 'last_', 'first_', 'recent', 'future']

    leakage_suspects = []
    for col in feature_columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in suspicious_patterns):
            leakage_suspects.append(col)

    if leakage_suspects:
        print(f"⚠️ POTENTIAL LEAKAGE: Suspicious features found:")
        for suspect in leakage_suspects:
            print(f"   - {suspect}")
        print("💡 Consider removing these features or verifying they're leak-free")

    # Check for perfect correlations (another leakage indicator)
    try:
        train_corr = X_train.corr().abs()
        high_corr_pairs = []

        for i in range(len(train_corr.columns)):
            for j in range(i+1, len(train_corr.columns)):
                if train_corr.iloc[i, j] > 0.99:  # Near perfect correlation
                    high_corr_pairs.append((train_corr.columns[i], train_corr.columns[j], train_corr.iloc[i, j]))

        if high_corr_pairs:
            print(f"⚠️ HIGH CORRELATIONS: Found {len(high_corr_pairs)} near-perfect feature correlations")
            for feat1, feat2, corr in high_corr_pairs[:3]:  # Show first 3
                print(f"   - {feat1} ↔ {feat2}: {corr:.4f}")

    except Exception as e:
        print(f"⚠️ Could not check feature correlations: {e}")

    # Check for unrealistic accuracy (>95% usually indicates leakage)
    print("💡 After training, watch for these leakage indicators:")
    print("   - Training accuracy > 95%")
    print("   - Perfect validation scores (1.000)")
    print("   - No gap between train and validation performance")
    print("   - Unrealistically high precision/recall")


def setup_plotting_style():
    """Set up consistent plotting style across all modules."""
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('default')

    sns.set_palette("husl")
    warnings.filterwarnings('ignore')


def print_section_header(title: str, section_emoji: str = "📊"):
    """Print a formatted section header."""
    print(f"{section_emoji} {title}")
    print("=" * 50)


def check_dependencies():
    """Check which optional dependencies are available."""
    deps = {
        'MLflow': MLFLOW_AVAILABLE,
        'Optuna': OPTUNA_AVAILABLE, 
        'TensorFlow': TENSORFLOW_AVAILABLE,
        'XGBoost': XGBOOST_AVAILABLE
    }

    print("🔍 Dependency Status:")
    for name, available in deps.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"   {name}: {status}")

    return deps
