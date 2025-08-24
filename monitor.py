"""
Runtime Monitoring (Performance & Data Drift)

Computes performance against labeled fresh data and detects distribution
drift (KS test fallback to mean/std shift). Produces a combined report and
a boolean recommendation to retrain.

Author: Ahmed Alghaith
Date: August 2025
"""
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import pandas as pd

class ChurnModelMonitoringSystem:
    """Monitor model health over time (concept/data drift signals).

    Args:
        model: Trained estimator.
        baseline_performance: Dict of baseline metrics from last accepted model.
        feature_columns: Feature names expected by the model.

    Attributes:
        performance_threshold: Allowed absolute drop in metrics (default 0.05).
        drift_threshold: Fraction of features allowed to drift (default 0.20).
    """

    def __init__(self, model, baseline_performance: dict, feature_columns: list):
        """
        Initialize monitoring system.
        Args:
            model: Trained ML model
            baseline_performance: {'accuracy','precision','recall','f1','roc_auc'}
            feature_columns: List of feature column names
        """
        self.model = model
        self.baseline_performance = baseline_performance
        self.feature_columns = feature_columns
        self.performance_threshold = 0.05    # 5% drop allowed
        self.drift_threshold = 0.20         # 20% of features drifted

    def evaluate_model_drift(self, X_new: pd.DataFrame, y_new: pd.Series) -> dict:
        """
        Compare current metrics vs baseline; flag significant drops.

        Returns:
            Dict containing current metrics, per-metric drops, and a boolean
            `retraining_needed`.
        """
        print("🔍 Evaluating model performance drift...")
        try:
            y_pred = self.model.predict(X_new)
            metrics = {
                'accuracy': accuracy_score(y_new, y_pred),
                'precision': precision_score(y_new, y_pred, zero_division=0),
                'recall': recall_score(y_new, y_pred, zero_division=0),
                'f1': f1_score(y_new, y_pred, zero_division=0)
            }
            if hasattr(self.model, 'predict_proba') and len(np.unique(y_new)) > 1:
                proba = self.model.predict_proba(X_new)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_new, proba)
            else:
                metrics['roc_auc'] = None

            drops = {
                m: self.baseline_performance[m] - metrics[m]
                for m in metrics if self.baseline_performance.get(m) is not None
            }
            significant = {m: d for m, d in drops.items() if d > self.performance_threshold}
            return {
                'current_performance': metrics,
                'performance_drops': drops,
                'significant_drops': significant,
                'retraining_needed': bool(significant),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Drift evaluation failed: {e}")
            return {
                'error': str(e),
                'retraining_needed': True,
                'timestamp': datetime.now().isoformat()
            }

    def detect_data_drift(self, X_orig: pd.DataFrame, X_new: pd.DataFrame) -> dict:
        """
        Full report combining performance and data drift with a recommendation.

        Returns:
            Dict with `performance_analysis`, optional `data_drift_analysis`,
            and `recommendation.retrain` boolean.
        """
        print("📊 Detecting data drift in features...")
        drifted = []
        results = {}
        try:
            from scipy.stats import ks_2samp
            for f in self.feature_columns:
                if f in X_orig and f in X_new:
                    stat, p = ks_2samp(X_orig[f].dropna(), X_new[f].dropna())
                    is_drift = p < 0.05
                    results[f] = {'ks_stat': stat, 'p_value': p, 'drift': is_drift}
                    if is_drift:
                        drifted.append(f)
        except ImportError:
            # fallback: mean/std shift
            for f in self.feature_columns:
                if f in X_orig and f in X_new:
                    mu0, mu1 = X_orig[f].mean(), X_new[f].mean()
                    sd0 = X_orig[f].std() or 1.0
                    score = abs(mu1 - mu0) / sd0
                    is_drift = score > 2.0
                    results[f] = {'drift_score': score, 'drift': is_drift}
                    if is_drift:
                        drifted.append(f)

        summary = {
            'total_features': len(results),
            'drifted_features': drifted,
            'drift_count': len(drifted),
            'drift_ratio': len(drifted) / max(1, len(results)),
            'significant_drift': len(drifted) / max(1, len(results)) > self.drift_threshold,
            'detailed': results
        }
        return summary

    def generate_monitoring_report(self, X_new: pd.DataFrame, y_new: pd.Series,
                                   X_orig: pd.DataFrame = None) -> dict:
        """
        Full monitoring report combining performance and data drift.
        """
        perf = self.evaluate_model_drift(X_new, y_new)
        report = {'performance_analysis': perf, 'timestamp': datetime.now().isoformat()}
        if X_orig is not None:
            drift = self.detect_data_drift(X_orig, X_new)
            report['data_drift_analysis'] = drift
        else:
            report['data_drift_analysis'] = {'note': 'No baseline provided'}
        report['recommendation'] = {
            'retrain': perf.get('retraining_needed', False) or report['data_drift_analysis']['significant_drift']
        }
        return report
