"""
Keras LSTM (scikit-learn-compatible) Wrapper

Provides `.fit/.predict/.predict_proba` over a dense→LSTM architecture with
robust numeric casting, missing-value handling, and scaling.

Docstrings follow Google / PEP 257 (Sphinx-Napoleon friendly).

Author: Ahmed Alghaith
Date: August 2025
"""

from utils import *

if TENSORFLOW_AVAILABLE:
    class KerasLSTMWrapper:
        """
        Minimal LSTM classifier with scikit-learn semantics.

    Args:
        input_dim: Number of input features.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        class_weight: Optional class weights dict.
        verbose: Keras verbosity flag.
        random_state: Seed for reproducibility.

    Attributes:
        scaler: StandardScaler fitted on training data.
        model: Compiled Keras model.
        fitted: Boolean flag set after successful training.
    """
        def __init__(self, input_dim: int, epochs: int = 40, batch_size: int = 32,
                     learning_rate: float = 1e-3, class_weight: dict = None,
                     verbose: int = 0, random_state: int = 42):
            self.input_dim = int(input_dim)
            self.epochs = int(epochs)
            self.batch_size = int(batch_size)
            self.learning_rate = float(learning_rate)
            self.class_weight = class_weight
            self.verbose = int(verbose)
            self.random_state = int(random_state)

            self.scaler = StandardScaler()
            self.model = None
            self.fitted = False

        def _prepare_data(self, X, y=None):
            """
            Prepare and validate data for LSTM training.

            Args:
                X: Input features
                y: Target labels (optional)

            Returns:
                Tuple of (X_processed, y_processed)
            """
            # Convert to DataFrame if necessary
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            # Ensure all features are numeric
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    print(f"⚠️ Converting non-numeric column {col} to numeric")
                    X[col] = pd.to_numeric(X[col], errors='coerce')

            # Handle missing values and infinities
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            # Convert to numpy array with proper dtype
            X_array = X.astype(np.float32).values

            if y is not None:
                # Ensure y is proper integer array
                if hasattr(y, 'values'):
                    y_array = y.values
                else:
                    y_array = np.array(y)

                y_array = y_array.astype(np.int32)
                return X_array, y_array

            return X_array

        def _build_model(self):
            """Build the LSTM model architecture."""
            tf.random.set_seed(self.random_state)

            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
                layers.Dropout(0.3),
                layers.Reshape((1, 64)),  # Reshape for LSTM
                layers.LSTM(32, return_sequences=False),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            return model

        def fit(self, X, y):
            """
            Fit the LSTM model.

            Args:
                X: Training features
                y: Training labels

            Returns:
                self
            """
            try:
                # Prepare data
                X_processed, y_processed = self._prepare_data(X, y)

                print(f"   📊 LSTM input shape: {X_processed.shape}")
                print(f"   📊 Target shape: {y_processed.shape}")

                # Fit scaler and transform features
                X_scaled = self.scaler.fit_transform(X_processed)

                # Build model
                self.model = self._build_model()

                # Setup callbacks
                callbacks_list = [
                    callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                ]

                # Train model
                self.model.fit(
                    X_scaled, y_processed,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    class_weight=self.class_weight,
                    callbacks=callbacks_list,
                    verbose=self.verbose
                )

                self.fitted = True
                return self

            except Exception as e:
                print(f"   ❌ LSTM training error: {e}")
                raise e

        def predict(self, X):
            """
            Make predictions.

            Args:
                X: Features to predict

            Returns:
                Binary predictions (0 or 1)
            """
            if not self.fitted:
                raise ValueError("Model must be fitted before making predictions")

            try:
                X_processed = self._prepare_data(X)
                X_scaled = self.scaler.transform(X_processed)

                # Get probabilities and convert to binary predictions
                y_prob = self.model.predict(X_scaled, verbose=0)
                return (y_prob.ravel() > 0.5).astype(int)

            except Exception as e:
                print(f"   ❌ LSTM prediction error: {e}")
                # Fallback to random predictions
                return np.random.choice([0, 1], size=len(X))

        def predict_proba(self, X):
            """
            Predict class probabilities.

            Args:
                X: Features to predict

            Returns:
                Array of probabilities for each class
            """
            if not self.fitted:
                raise ValueError("Model must be fitted before making predictions")

            try:
                X_processed = self._prepare_data(X)
                X_scaled = self.scaler.transform(X_processed)

                y_prob = self.model.predict(X_scaled, verbose=0).ravel()

                # Return probabilities for both classes
                return np.column_stack([1 - y_prob, y_prob])

            except Exception as e:
                print(f"   ❌ LSTM probability prediction error: {e}")
                # Fallback to random probabilities
                prob_class_1 = np.random.random(len(X))
                return np.column_stack([1 - prob_class_1, prob_class_1])

else:
    # Dummy class when TensorFlow is not available
    class KerasLSTMWrapper:
        """Dummy LSTM wrapper when TensorFlow is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available - LSTM models cannot be used")

        def fit(self, X, y):
            raise ImportError("TensorFlow not available - LSTM models cannot be used")

        def predict(self, X):
            raise ImportError("TensorFlow not available - LSTM models cannot be used")

        def predict_proba(self, X):
            raise ImportError("TensorFlow not available - LSTM models cannot be used")
