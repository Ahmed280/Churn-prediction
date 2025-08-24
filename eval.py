"""
Evaluation & Hyperparameter Tuning

- Unified evaluation metrics (accuracy, precision, recall, F1, ROC-AUC).
- Multi-model training summary.
- Robust tuning: Optuna (child runs in MLflow) with grid-search fallback.

Author: Ahmed Alghaith
Date: August 2025
"""

from utils import *
import mlflow
import optuna

def evaluate_churn_model(model, X_test, y_test, model_name: str = "Model"):
    """
    Compute core metrics + classification report for a trained model.

    Args:
        model: Fitted estimator with `predict` (and optionally `predict_proba`).
        X_test: Test features.
        y_test: Test labels.
        model_name: Label for logs/prints.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = y_pred
    else:
        y_pred_proba = y_pred

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    print(f"🎯 {model_name} Evaluation Results:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    # Classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

    return metrics


def train_and_evaluate_models(models_config, X_val, y_val):
    """
    Fit each configured model; summarize train/val metrics and successes.

    Args:
        models_config: Mapping model_name → {"model": estimator, "data": (X, y)}.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        (trained_models, results_df): dict + DataFrame of per-model results.
    """
    trained_models = {}
    results = []

    print("🤖 Training multiple machine learning models...\n")

    for model_name, config in models_config.items():
        model = config["model"]
        X_train_data, y_train_data = config["data"]

        try:
            print(f"🔧 Training {model_name}...")
            print(f"   Training on {len(y_train_data)} samples with {len(np.unique(y_train_data))} classes")

            # Train the model
            model.fit(X_train_data, y_train_data)

            # Evaluate on training and validation sets
            train_pred = model.predict(X_train_data)
            val_pred = model.predict(X_val)

            train_acc = accuracy_score(y_train_data, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, zero_division=0)
            val_recall = recall_score(y_val, val_pred, zero_division=0)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)

            print("   ✅ Training successful!")
            print(f"   Train Accuracy: {train_acc:.3f}")
            print(f"   Val Accuracy: {val_acc:.3f}")
            print(f"   Val F1: {val_f1:.3f}")

            # Check for overfitting
            if train_acc - val_acc > 0.15:
                gap = train_acc - val_acc
                print(f"   ⚠️ Potential overfitting detected (gap: {gap:.3f})")

            results.append({
                "Model": model_name,
                "Status": "Success",
                "Train_Accuracy": train_acc,
                "Val_Accuracy": val_acc,
                "Val_Precision": val_precision,
                "Val_Recall": val_recall,
                "Val_F1": val_f1,
                "Training_Samples": len(y_train_data),
                "Classes_in_Training": len(np.unique(y_train_data))
            })

            trained_models[model_name] = model

        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            results.append({
                "Model": model_name,
                "Status": f"Failed: {e}",
                "Train_Accuracy": None,
                "Val_Accuracy": None,
                "Val_Precision": None,
                "Val_Recall": None,
                "Val_F1": None,
                "Training_Samples": len(y_train_data) if 'y_train_data' in locals() else 0,
                "Classes_in_Training": len(np.unique(y_train_data)) if 'y_train_data' in locals() else 0
            })

        print("")  # Add spacing between models

    # Compile results
    results_df = pd.DataFrame(results)

    print("📊 Training Results Summary:")
    print(f"Models attempted: {len(results_df)}")
    print(f"Successful trainings: {len(results_df[results_df['Status']=='Success'])}")
    print(f"Failed trainings: {len(results_df[results_df['Status']!='Success'])}\n")

    return trained_models, results_df


def get_model_param_space(model_type):
    """
    Lightweight grid search with F1 on validation.

    Args:
        model_class: Estimator class.
        param_space: Dict of param ranges (lists).
        X_train, y_train: Train set.
        X_val, y_val: Validation set.
        model_name: For logs.

    Returns:
        (best_model, best_score, best_params).
    """
    param_spaces = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [200, 500, 1000]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'decision_tree': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    }

    return param_spaces.get(model_type, {})


def grid_search_hyperparameters(model_class, param_space, X_train, y_train, X_val, y_val, model_name="Model"):
    """
    Optuna tuning with **nested MLflow child runs** per trial.

    Args:
        model_class: Estimator class.
        model_type: String key for parameter space logic.
        X_train, y_train, X_val, y_val: Datasets.
        n_trials: Number of trials.

    Returns:
        (best_model, best_score, best_params).
    """
    from itertools import product

    print(f"🔍 Grid search hyperparameter tuning for {model_name}...")

    # Generate all parameter combinations
    param_names = list(param_space.keys())
    param_values = list(param_space.values())

    best_score = -1
    best_params = None
    best_model = None

    # Limit combinations to avoid excessive computation
    max_combinations = 50
    all_combinations = list(product(*param_values))

    if len(all_combinations) > max_combinations:
        print(f"⚠️ Too many combinations ({len(all_combinations)}), sampling {max_combinations}")
        import random
        combinations = random.sample(all_combinations, max_combinations)
    else:
        combinations = all_combinations

    print(f"🔧 Testing {len(combinations)} parameter combinations...")

    for i, param_combo in enumerate(combinations):
        try:
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            params['random_state'] = 42  # Ensure reproducibility

            # Create and train model
            model = model_class(**params)
            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            if f1 > best_score:
                best_score = f1
                best_params = params.copy()
                best_model = model

            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(combinations)} combinations tested")

        except Exception as e:
            # Skip invalid parameter combinations
            continue

    print(f"✅ Best F1 score: {best_score:.4f}")
    print(f"✅ Best parameters: {best_params}")

    return best_model, best_score, best_params


def optuna_hyperparameter_tuning(model_class, model_type, X_train, y_train, X_val, y_val, n_trials=50):
    """
    Optuna-based hyperparameter tuning with per-trial MLflow nested runs.

    Args:
        model_class: Estimator class (e.g., RandomForestClassifier)
        model_type:  One of {'random_forest','gradient_boosting','xgboost','logistic_regression',...}
        X_train, y_train: Training data
        X_val, y_val:     Validation data
        n_trials:         Number of optimization trials

    Returns:
        (best_model, best_score, best_params)
    """
    # ---- Robust Optuna availability check ----
    if not globals().get("OPTUNA_AVAILABLE", False):
        raise ImportError("Optuna not available")


    # Ensure y is 1-D
    try:
        y_train_ = y_train.ravel() if hasattr(y_train, "ravel") else y_train
        y_val_   = y_val.ravel() if hasattr(y_val, "ravel") else y_val
    except Exception:
        y_train_, y_val_ = y_train, y_val

    def objective(trial):
        # ----- Suggest params per model_type -----
        if model_type == "random_forest":
            params = {
                "n_estimators":       trial.suggest_int("n_estimators", 50, 400),
                "max_depth":          trial.suggest_int("max_depth", 3, 20),
                "min_samples_split":  trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":   trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state":       42,
                "n_jobs":             -1,
            }

        elif model_type == "gradient_boosting":
            params = {
                "n_estimators":       trial.suggest_int("n_estimators", 50, 400),
                "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth":          trial.suggest_int("max_depth", 2, 10),
                "min_samples_split":  trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":   trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state":       42,
            }

        elif model_type == "xgboost":
            # Works with xgboost.sklearn.XGBClassifier
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 50, 600),
                "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth":         trial.suggest_int("max_depth", 2, 10),
                "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 10.0),
                "random_state":      42,
                "eval_metric":       "logloss",
                "n_jobs":            -1,
                "verbosity":         0,
                "use_label_encoder": False,
            }

        elif model_type == "logistic_regression":
            params = {
                "C":         trial.suggest_float("C", 1e-2, 1e2, log=True),
                "solver":    trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
                "max_iter":  trial.suggest_int("max_iter", 100, 2000),
                "random_state": 42,
            }

        else:
            # Fallback with only random_state for unknown types
            params = {"random_state": 42}

        # ----- Train & evaluate one trial -----
        # Make the trial an MLflow *nested* run so it never collides with the parent
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)

            model = model_class(**params)
            model.fit(X_train, y_train_)
            y_pred = model.predict(X_val)
            val_f1 = f1_score(y_val_, y_pred, zero_division=0)

            # Log AUC if predict_proba available
            try:
                proba = getattr(model, "predict_proba")(X_val)[:, 1]
                val_auc = roc_auc_score(y_val_, proba)
                mlflow.log_metric("val_auc", float(val_auc))
            except Exception:
                val_auc = None  # not all models support predict_proba

            mlflow.log_metric("val_f1", float(val_f1))

            # Optionally store the trial number explicitly
            mlflow.log_param("trial_number", trial.number)

        return val_f1

    # ----- Run study -----
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials))

    best_params = study.best_params
    best_score  = float(study.best_value)

    # ----- Refit best model on full training set -----
    best_model = model_class(**best_params) if best_params else model_class()
    best_model.fit(X_train, y_train_)

    return best_model, best_score, best_params




def robust_hyperparameter_tuning(model_configs, X_train, y_train, X_val, y_val, use_optuna=True):
    """
    Robust hyperparameter tuning on a single model (or multiple)
    but typically invoked with a single best-model config.
    Logs parameters and metrics to MLflow.
    """
    print("🚀 Starting robust hyperparameter tuning...")
    mlflow.set_experiment("churn_model_tuning")
    # Ensure only one config is provided
    if len(model_configs) != 1:
        raise ValueError("Expected exactly one model_config entry for best-model tuning")

    model_name, (model_class, model_type) = next(iter(model_configs.items()))
    print(f"\n🔧 Tuning {model_name}...")

    with mlflow.start_run(run_name=f"tune_{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_type)
        best_model, best_score, best_params = None, None, None

        # Attempt Optuna first
        try:
            if use_optuna and OPTUNA_AVAILABLE:
                best_model, best_score, best_params = optuna_hyperparameter_tuning(
                    model_class, model_type, X_train, y_train, X_val, y_val
                )
                method = "Optuna"
            else:
                raise ImportError("Skipping Optuna")
        except Exception as optuna_error:
            print(f" ⚠️ Optuna failed ({optuna_error}), falling back to grid search")
            mlflow.log_warning(str(optuna_error))
            param_space = get_model_param_space(model_type)
            if param_space:
                best_model, best_score, best_params = grid_search_hyperparameters(
                    model_class, param_space, X_train, y_train, X_val, y_val, model_name
                )
                method = "Grid Search"
            else:
                print(" ⚠️ No parameter space defined, training with default parameters")
                best_model = model_class(random_state=42)
                best_model.fit(X_train, y_train)
                val_pred = best_model.predict(X_val)
                best_score = f1_score(y_val, val_pred, zero_division=0)
                best_params = {"random_state": 42}
                method = "Default"

        print(f"   ✅ {model_name} tuned using {method}")
        print(f"   📊 Best F1 Score: {best_score:.4f}")

        # Log to MLflow
        mlflow.log_param("tuning_method", method)
        for p_name, p_val in best_params.items():
            mlflow.log_param(p_name, p_val)
        mlflow.log_metric("best_val_f1", best_score)

        # Save the best model artifact
        mlflow.sklearn.log_model(best_model, artifact_path="model")

    print(f"\n🏆 Completed tuning for: {model_name}")
    results = {
        model_name: {
            "model": best_model,
            "score": best_score,
            "params": best_params,
            "method": method
        }
    }
    return results, best_model




# Legacy function for backward compatibility
def hyperparameter_tuning_with_optuna(X_train, y_train, X_val, y_val):
    """
    Legacy function - now uses robust hyperparameter tuning.
    """
    print("🔄 Using robust hyperparameter tuning...")
    results, best_model = robust_hyperparameter_tuning(
        None, X_train, y_train, X_val, y_val, use_optuna=True
    )
    return best_model, 0.5  # Return model and default threshold
