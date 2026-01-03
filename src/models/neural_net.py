"""
Neural network trigger model.

Deep learning approach for trigger decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks


class NeuralNetworkTrigger:
    """Deep neural network for trigger decisions."""
    
    def __init__(self, config: Dict):
        """
        Initialize neural network trigger.
        
        Args:
            config: Configuration dictionary with neural_net parameters
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.scaler = None
        self.history = None
    
    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build neural network architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        architecture = self.config['architecture']
        dropout = self.config['dropout']
        use_batch_norm = self.config['batch_norm']
        
        model = models.Sequential(name='TriggerNN')
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,), name='input'))
        
        # Hidden layers
        for i, units in enumerate(architecture[:-1]):
            model.add(layers.Dense(
                units,
                activation=self.config['activation'],
                name=f'dense_{i+1}'
            ))
            
            if use_batch_norm:
                model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            
            if dropout > 0:
                model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(
            1,
            activation='sigmoid',
            name='output'
        ))
        
        # Compile model
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None
    ) -> Dict:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history dictionary
        """
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Build model
        self.model = self.build_model(X_train.shape[1])
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train_scaled,
            y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            validation_split=self.config['validation_split'] if validation_data is None else 0,
            callbacks=callback_list,
            verbose=1
        )
        
        self.is_trained = True
        
        return self.history.history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: DataFrame with event features
            
        Returns:
            Binary predictions (1 = accept, 0 = reject)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict(X_scaled, verbose=0)
        
        return (proba >= 0.5).astype(int).flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class probabilities.
        
        Args:
            X: DataFrame with event features
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        proba_positive = self.model.predict(X_scaled, verbose=0).flatten()
        
        proba = np.zeros((len(proba_positive), 2))
        proba[:, 1] = proba_positive
        proba[:, 0] = 1 - proba_positive
        
        return proba
    
    def get_latency(self, X: pd.DataFrame, n_iterations: int = 10000) -> float:
        """
        Measure inference latency per event.
        
        Args:
            X: DataFrame with event features
            n_iterations: Number of timing iterations
            
        Returns:
            Average latency in microseconds
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before benchmarking")
        
        X_scaled = self.scaler.transform(X)
        
        # Warm-up
        for _ in range(100):
            _ = self.model.predict(X_scaled[:100], verbose=0)
        
        # Benchmark single-event processing
        times = []
        for _ in range(n_iterations):
            idx = np.random.randint(0, len(X_scaled))
            event = X_scaled[idx:idx+1]
            
            start = time.perf_counter()
            _ = self.model.predict(event, verbose=0)
            end = time.perf_counter()
            
            times.append((end - start) * 1e6)  # Convert to microseconds
        
        return np.mean(times)
    
    def save(self, filepath: str):
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        self.model.save(filepath)
        
        # Save scaler separately
        import pickle
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, filepath: str):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)
        
        # Load scaler
        import pickle
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.is_trained = True
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet")
    
    def plot_training_history(self):
        """Plot training curves."""
        if self.history is None:
            print("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        history_dict = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history_dict['loss'], label='Training')
        if 'val_loss' in history_dict:
            axes[0, 0].plot(history_dict['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history_dict['accuracy'], label='Training')
        if 'val_accuracy' in history_dict:
            axes[0, 1].plot(history_dict['val_accuracy'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(history_dict['auc'], label='Training')
        if 'val_auc' in history_dict:
            axes[1, 0].plot(history_dict['val_auc'], label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('Training and Validation AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision & Recall
        axes[1, 1].plot(history_dict['precision'], label='Precision')
        axes[1, 1].plot(history_dict['recall'], label='Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig