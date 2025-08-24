"""
Temporal Splitting, Validation, and Imbalance Handling (Leak-Safe)

Provides:
- Temporal splits (train/val/test) aligned to business time.
- Strict anti-leakage checks (feature type, names, datetimes).
- Class imbalance strategies (class weights + balanced sampling).
- Final numeric-only, NaN/Inf-free datasets for modeling.

Conforms to Google-style docstrings / PEP 257.

Author: Ahmed Alghaith
Date: August 2025
"""

from utils import *

def validate_features_for_ml(X, feature_name="Features"):
    """
    Ensure ML-compatibility: numeric dtypes, no NaN/Inf, remove datetime.

    Args:
        X: Feature matrix as DataFrame/array-like.
        feature_name: Label used in logs/messages.

    Returns:
        Cleaned numeric DataFrame safe for model ingestion.
    """
    print(f"🔍 Validating {feature_name} for ML compatibility...")

    # Convert to DataFrame if necessary
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    original_shape = X.shape

    # Check for non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"   ⚠️ Found non-numeric columns: {non_numeric_cols}")

        for col in non_numeric_cols:
            if X[col].dtype == 'datetime64[ns]':
                print(f"   🚨 LEAKAGE ALERT: Removing datetime column {col}")
                X = X.drop(col, axis=1)
            elif X[col].dtype == 'object':
                # Try to convert object columns to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce')
                print(f"   🔢 Attempted to convert object column {col} to numeric")

    # Handle missing values
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        X = X.fillna(0)
        print(f"   🔧 Filled {missing_before} missing values with 0")

    # Handle infinite values
    inf_before = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    if inf_before > 0:
        X = X.replace([np.inf, -np.inf], 0)
        print(f"   ♾️ Replaced {inf_before} infinite values with 0")

    # Ensure all columns are numeric
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(0)

    # Final validation
    final_shape = X.shape
    print(f"   ✅ {feature_name} validation complete: {original_shape} -> {final_shape}")
    print(f"   🔢 All {len(X.columns)} columns are now numeric")

    return X


def detect_leakage_indicators(y_train_pred, y_train_true, y_val_pred, y_val_true, model_name="Model"):
    """
    Heuristics to flag suspicious performance patterns (possible leakage).

    Args:
        y_train_pred: Train predictions.
        y_train_true: Train labels.
        y_val_pred: Validation predictions.
        y_val_true: Validation labels.
        model_name: Name used for printed diagnostics.

    Returns:
        True if leakage is suspected; False otherwise.
    """
    train_acc = accuracy_score(y_train_true, y_train_pred)
    val_acc = accuracy_score(y_val_true, y_val_pred)
    train_f1 = f1_score(y_train_true, y_train_pred, zero_division=0)
    val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)

    print(f"🔍 Leakage Detection for {model_name}:")

    leakage_detected = False

    # Check for perfect or near-perfect scores
    if train_acc >= 0.98 or val_acc >= 0.98:
        print(f"   🚨 LEAKAGE ALERT: Suspiciously high accuracy (Train: {train_acc:.3f}, Val: {val_acc:.3f})")
        leakage_detected = True

    if train_f1 >= 0.98 or val_f1 >= 0.98:
        print(f"   🚨 LEAKAGE ALERT: Suspiciously high F1 score (Train: {train_f1:.3f}, Val: {val_f1:.3f})")
        leakage_detected = True

    # Check for unrealistic generalization (validation better than training)
    if val_acc > train_acc + 0.02:
        print(f"   🚨 LEAKAGE ALERT: Validation accuracy higher than training ({val_acc:.3f} > {train_acc:.3f})")
        leakage_detected = True

    # Check for lack of overfitting (suspicious for complex models)
    generalization_gap = train_acc - val_acc
    if generalization_gap < 0.01 and train_acc > 0.8:
        print(f"   ⚠️ SUSPICIOUS: No generalization gap (gap: {generalization_gap:.3f})")
        leakage_detected = True

    if not leakage_detected:
        print(f"   ✅ No obvious leakage indicators detected")
        print(f"      Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Gap: {generalization_gap:.3f}")

    return leakage_detected


def handle_class_imbalance(X_train, y_train, strategy='both'):
    """
    Create class weights and/or balanced resamples (down/up) for training.

    Args:
        X_train: Training features (numeric).
        y_train: Training labels (array-like).
        strategy: 'weights', 'balance', or 'both'.

    Returns:
        Dict containing class weights and/or balanced (X, y) copies.
    """
    print("⚖️ Handling class imbalance with leakage prevention...")

    # Validate inputs first
    X_train = validate_features_for_ml(X_train, "Training Features")

    # Check for leakage indicators in feature names
    validate_no_leakage(X_train, X_train, X_train, X_train.columns.tolist())

    # Ensure y_train is proper integer array
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values
    else:
        y_train_array = np.array(y_train)

    y_train_array = y_train_array.astype(int)

    # Analyze current training set
    unique_classes_in_data = np.unique(y_train_array)
    print(f"\n🔍 Training set analysis:")
    print(f"Classes present: {unique_classes_in_data}")
    print(f"Total samples: {len(y_train_array)}")

    class_counts = np.bincount(y_train_array)
    print(f"Class distribution: {dict(enumerate(class_counts))}")

    # Check if we have enough samples and both classes
    if len(unique_classes_in_data) < 2:
        print("❌ ERROR: Need at least 2 classes for classification")
        return None

    if len(y_train_array) < 10:
        print("❌ ERROR: Too few samples for reliable training")
        return None

    results = {}

    # Strategy 1: Class weights
    if strategy in ['weights', 'both']:
        try:
            class_weight_array = compute_class_weight(
                'balanced', 
                classes=unique_classes_in_data, 
                y=y_train_array
            )
            class_weight_dict = dict(zip(unique_classes_in_data, class_weight_array))
            results['class_weights'] = class_weight_dict
            results['X_train_weighted'] = X_train
            results['y_train_weighted'] = pd.Series(y_train_array)

            print(f"✅ Computed class weights: {class_weight_dict}")

        except Exception as e:
            print(f"❌ Failed to compute class weights: {e}")
            results['class_weights'] = None

    # Strategy 2: Balanced sampling  
    if strategy in ['balance', 'both']:
        try:
            # Find minority class
            minority_class = np.argmin(class_counts)
            minority_count = class_counts[minority_class]

            if minority_count == 0:
                print("❌ Cannot balance: minority class has 0 samples")
                results['balanced_data'] = None
            else:
                # Sample equal amounts from each class
                X_balanced_list = []
                y_balanced_list = []

                for class_label in unique_classes_in_data:
                    class_indices = np.where(y_train_array == class_label)[0]

                    if len(class_indices) >= minority_count:
                        # Downsample majority class
                        selected_indices = np.random.choice(
                            class_indices, 
                            size=minority_count, 
                            replace=False
                        )
                    else:
                        # Upsample minority class (with replacement)
                        selected_indices = np.random.choice(
                            class_indices, 
                            size=minority_count, 
                            replace=True
                        )

                    X_balanced_list.append(X_train.iloc[selected_indices])
                    y_balanced_list.append(pd.Series(y_train_array[selected_indices]))

                X_train_balanced = pd.concat(X_balanced_list, ignore_index=True)
                y_train_balanced = pd.concat(y_balanced_list, ignore_index=True)

                # Shuffle the balanced data
                shuffle_idx = np.random.permutation(len(X_train_balanced))
                X_train_balanced = X_train_balanced.iloc[shuffle_idx].reset_index(drop=True)
                y_train_balanced = y_train_balanced.iloc[shuffle_idx].reset_index(drop=True)

                # Validate balanced data
                X_train_balanced = validate_features_for_ml(X_train_balanced, "Balanced Training Features")

                results['X_train_balanced'] = X_train_balanced
                results['y_train_balanced'] = y_train_balanced

                print(f"✅ Created balanced dataset:")
                print(f"   Original: {len(y_train_array)} samples")
                print(f"   Balanced: {len(y_train_balanced)} samples")
                print(f"   New distribution: {dict(enumerate(np.bincount(y_train_balanced)))}")

        except Exception as e:
            print(f"❌ Failed to create balanced dataset: {e}")
            results['balanced_data'] = None

    return results


def prepare_training_data(user_features_df):
    """
    Produce leak-safe splits + validated, numeric features.

    Args:
        user_features_df: Feature table with `userId` and `churn`.

    Returns:
        Dict: X_train, y_train, X_val, y_val, X_test, y_test, feature_columns,
        and (optional) imbalance artifacts.
    """
    print("🔄 Preparing training data with leakage prevention...")

    # Temporal split
    train_df, val_df, test_df = temporal_split(user_features_df)

    # Identify feature columns (exclude userId and churn)
    feature_columns = [col for col in user_features_df.columns if col not in ['userId', 'churn']]

    # Validate feature names for leakage indicators
    print("\n🔍 Checking feature names for leakage indicators...")
    validate_no_leakage(train_df[feature_columns], val_df[feature_columns], test_df[feature_columns], feature_columns)

    # Clean splits with validation
    X_train, y_train = clean_split(train_df[feature_columns], train_df['churn'])
    X_val, y_val = clean_split(val_df[feature_columns], val_df['churn'])
    X_test, y_test = clean_split(test_df[feature_columns], test_df['churn'])

    # Validate all feature sets
    X_train = validate_features_for_ml(X_train, "Training Features")
    X_val = validate_features_for_ml(X_val, "Validation Features")
    X_test = validate_features_for_ml(X_test, "Test Features")

    print(f"\n✅ Data splits prepared and validated:")
    print(f"   Train: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"   Validation: {len(X_val)} samples, {len(X_val.columns)} features") 
    print(f"   Test: {len(X_test)} samples, {len(X_test.columns)} features")

    # Handle class imbalance
    imbalance_results = handle_class_imbalance(X_train, y_train, strategy='both')

    # Prepare final results dictionary
    results = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_columns': list(X_train.columns)  # Updated feature columns after cleaning
    }

    # Add imbalance handling results
    if imbalance_results:
        results.update(imbalance_results)

    return results


def get_model_configurations(results_dict):
    """Build a suite of baseline model configs for comparison.

    Selects class-weighted vs balanced-data variants; adds XGBoost/LSTM if available.

    Args:
        results_dict: Output of `prepare_training_data()`.

    Returns:
        Mapping: model name → {"model": estimator, "data": (X, y)}.
    """
    X_train = results_dict['X_train']
    y_train = results_dict['y_train']
    class_weights = results_dict.get('class_weights', {})
    X_train_balanced = results_dict.get('X_train_balanced', X_train)
    y_train_balanced = results_dict.get('y_train_balanced', y_train)

    # Check for library availability
    lstm_available = TENSORFLOW_AVAILABLE if 'TENSORFLOW_AVAILABLE' in globals() else False
    xgb_available = XGBOOST_AVAILABLE if 'XGBOOST_AVAILABLE' in globals() else False

    models_config = {}

    # Always available sklearn models first
    models_config.update({
        "Random Forest (Class Weighted)": {
            "model": RandomForestClassifier(
                class_weight=class_weights if class_weights else 'balanced',
                random_state=42, 
                n_jobs=-1,
                n_estimators=50,  # Smaller for faster training
                max_depth=10      # Prevent overfitting
            ),
            "data": (X_train, y_train)
        },
        "Random Forest (Balanced Data)": {
            "model": RandomForestClassifier(
                random_state=42, 
                n_jobs=-1,
                n_estimators=50,
                max_depth=10
            ),
            "data": (X_train_balanced, y_train_balanced)
        },
        "Logistic Regression (Class Weighted)": {
            "model": LogisticRegression(
                class_weight=class_weights if class_weights else 'balanced',
                random_state=42, 
                n_jobs=-1, 
                max_iter=200, 
                solver='liblinear'
            ),
            "data": (X_train, y_train)
        },
        "Logistic Regression (Balanced Data)": {
            "model": LogisticRegression(
                random_state=42, 
                n_jobs=-1, 
                max_iter=200, 
                solver='liblinear'
            ),
            "data": (X_train_balanced, y_train_balanced)
        },
        "Gradient Boosting (Class Weighted)": {
            "model": GradientBoostingClassifier(
                random_state=42,
                n_estimators=50,   # Smaller for faster training
                max_depth=5        # Prevent overfitting
            ),
            "data": (X_train, y_train)
        },
        "Gradient Boosting (Balanced Data)": {
            "model": GradientBoostingClassifier(
                random_state=42,
                n_estimators=50,
                max_depth=5
            ),
            "data": (X_train_balanced, y_train_balanced)
        },
        "Decision Tree (Class Weighted)": {
            "model": DecisionTreeClassifier(
                class_weight=class_weights if class_weights else 'balanced',
                random_state=42,
                max_depth=8        # Prevent overfitting
            ),
            "data": (X_train, y_train)
        },
        "Decision Tree (Balanced Data)": {
            "model": DecisionTreeClassifier(
                random_state=42,
                max_depth=8
            ),
            "data": (X_train_balanced, y_train_balanced)
        }
    })

    # Add XGBoost models if available
    if xgb_available:
        try:
            scale_pos_weight = class_weights.get(1, 1.0) / class_weights.get(0, 1.0) if class_weights else 1.0

            models_config["XGBClassifier (Class Weighted)"] = {
                "model": XGBClassifier(
                    eval_metric='logloss',
                    random_state=42, 
                    n_jobs=-1, 
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=50,
                    max_depth=6,           # Prevent overfitting
                    learning_rate=0.1
                ),
                "data": (X_train, y_train)
            }

            models_config["XGBClassifier (Balanced Data)"] = {
                "model": XGBClassifier(
                    eval_metric='logloss', 
                    random_state=42, 
                    n_jobs=-1,
                    n_estimators=50,
                    max_depth=6,
                    learning_rate=0.1
                ),
                "data": (X_train_balanced, y_train_balanced)
            }
        except Exception as e:
            print(f"⚠️ XGBoost models skipped due to error: {e}")

    # Add LSTM models if available
    if lstm_available:
        try:
            from lstm_wrapper import KerasLSTMWrapper

            models_config["Simple LSTM (Class Weighted)"] = {
                "model": KerasLSTMWrapper(
                    input_dim=X_train.shape[1],
                    epochs=10,  # Reduced for faster training
                    batch_size=32,
                    learning_rate=1e-3,
                    class_weight=class_weights,
                    verbose=0,
                    random_state=42
                ),
                "data": (X_train, y_train)
            }

            models_config["Simple LSTM (Balanced Data)"] = {
                "model": KerasLSTMWrapper(
                    input_dim=X_train.shape[1],
                    epochs=10,
                    batch_size=32,
                    learning_rate=1e-3,
                    class_weight=None,
                    verbose=0,
                    random_state=42
                ),
                "data": (X_train_balanced, y_train_balanced)
            }
        except Exception as e:
            print(f"⚠️ LSTM models skipped due to error: {e}")

    print(f"✅ Created {len(models_config)} model configurations with leakage prevention")
    return models_config
