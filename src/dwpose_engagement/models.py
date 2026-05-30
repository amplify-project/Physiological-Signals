"""
Machine learning models for audience engagement prediction.

This module provides various ML models to predict engagement levels
from pose-derived features, including traditional ML and deep learning approaches.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass
import mlflow
import mlflow.sklearn
import mlflow.pytorch

logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Container for model training and evaluation results."""
    model: Any
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    metrics: Dict[str, float]
    feature_importance: Optional[np.ndarray]
    confusion_matrix: Optional[np.ndarray]
    model_type: str


class EngagementDataset(Dataset):
    """PyTorch Dataset for engagement prediction."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, scaler=None):
        """
        Initialize the dataset.
        
        Args:
            features: Feature matrix [N, F]
            targets: Target values [N]
            scaler: Optional feature scaler
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets.dtype != int else torch.LongTensor(targets)
        self.scaler = scaler
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class EngagementNN(nn.Module):
    """Neural network for engagement prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [256, 128, 64],
        output_dim: int = 1,
        dropout: float = 0.3,
        task_type: str = "regression"
    ):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            output_dim: Number of output classes/values
            dropout: Dropout probability
            task_type: "regression" or "classification"
        """
        super(EngagementNN, self).__init__()
        
        self.task_type = task_type
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if task_type == "classification" and output_dim > 1:
            layers.append(nn.Softmax(dim=1))
        elif task_type == "classification" and output_dim == 1:
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class EngagementPredictor:
    """
    Main class for training and evaluating engagement prediction models.
    
    Supports multiple model types including Random Forest, XGBoost, and Neural Networks.
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        task_type: str = "classification",
        config: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the engagement predictor.
        
        Args:
            model_type: Type of model ("random_forest", "xgboost", "neural_network")
            task_type: "classification" or "regression"
            config: Model-specific configuration
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.task_type = task_type
        self.config = config or {}
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if task_type == "classification" else None
        self.is_fitted = False
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specific model based on model_type."""
        if self.model_type == "random_forest":
            self._initialize_random_forest()
        elif self.model_type == "xgboost":
            self._initialize_xgboost()
        elif self.model_type == "neural_network":
            self._initialize_neural_network()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _initialize_random_forest(self):
        """Initialize Random Forest model."""
        rf_config = self.config.get("random_forest", {})
        
        if self.task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=rf_config.get("n_estimators", 100),
                max_depth=rf_config.get("max_depth", 10),
                min_samples_split=rf_config.get("min_samples_split", 5),
                min_samples_leaf=rf_config.get("min_samples_leaf", 2),
                random_state=self.random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=rf_config.get("n_estimators", 100),
                max_depth=rf_config.get("max_depth", 10),
                min_samples_split=rf_config.get("min_samples_split", 5),
                min_samples_leaf=rf_config.get("min_samples_leaf", 2),
                random_state=self.random_state
            )
    
    def _initialize_xgboost(self):
        """Initialize XGBoost model."""
        xgb_config = self.config.get("xgboost", {})
        
        if self.task_type == "classification":
            self.model = xgb.XGBClassifier(
                n_estimators=xgb_config.get("n_estimators", 100),
                max_depth=xgb_config.get("max_depth", 6),
                learning_rate=xgb_config.get("learning_rate", 0.1),
                subsample=xgb_config.get("subsample", 0.8),
                random_state=self.random_state
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=xgb_config.get("n_estimators", 100),
                max_depth=xgb_config.get("max_depth", 6),
                learning_rate=xgb_config.get("learning_rate", 0.1),
                subsample=xgb_config.get("subsample", 0.8),
                random_state=self.random_state
            )
    
    def _initialize_neural_network(self):
        """Initialize Neural Network model."""
        # Will be initialized in fit() when we know input dimensions
        self.nn_config = self.config.get("neural_network", {})
        self.model = None
    
    def prepare_data(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for training.
        
        Args:
            features: Feature matrix [N, F]
            targets: Target values [N]
            fit_scaler: Whether to fit the scaler
        
        Returns:
            Tuple of (scaled_features, encoded_targets)
        """
        # Scale features
        if fit_scaler:
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        
        # Encode targets for classification
        if self.task_type == "classification" and self.label_encoder:
            if fit_scaler:  # Only fit encoder during training
                encoded_targets = self.label_encoder.fit_transform(targets)
            else:
                encoded_targets = self.label_encoder.transform(targets)
        else:
            encoded_targets = targets
        
        return scaled_features, encoded_targets
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> ModelResults:
        """
        Train the engagement prediction model.
        
        Args:
            X_train: Training features [N, F]
            y_train: Training targets [N]
            X_val: Validation features [M, F] (optional)
            y_val: Validation targets [M] (optional)
        
        Returns:
            Training results
        """
        logger.info(f"Training {self.model_type} model for {self.task_type}")
        
        # Prepare training data
        X_train_scaled, y_train_encoded = self.prepare_data(X_train, y_train, fit_scaler=True)
        
        if self.model_type == "neural_network":
            return self._fit_neural_network(
                X_train_scaled, y_train_encoded, X_val, y_val
            )
        else:
            return self._fit_sklearn_model(X_train_scaled, y_train_encoded)
    
    def _fit_sklearn_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> ModelResults:
        """Fit sklearn-compatible model."""
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Make predictions on training data
        predictions = self.model.predict(X_train)
        
        # Get probabilities for classification
        probabilities = None
        if hasattr(self.model, "predict_proba") and self.task_type == "classification":
            probabilities = self.model.predict_proba(X_train)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, predictions, probabilities)
        
        # Get feature importance
        feature_importance = None
        if hasattr(self.model, "feature_importances_"):
            feature_importance = self.model.feature_importances_
        
        # Get confusion matrix for classification
        confusion_mat = None
        if self.task_type == "classification":
            confusion_mat = confusion_matrix(y_train, predictions)
        
        return ModelResults(
            model=self.model,
            predictions=predictions,
            probabilities=probabilities,
            metrics=metrics,
            feature_importance=feature_importance,
            confusion_matrix=confusion_mat,
            model_type=self.model_type
        )
    
    def _fit_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> ModelResults:
        """Fit neural network model."""
        # Initialize network
        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train)) if self.task_type == "classification" else 1
        
        self.model = EngagementNN(
            input_dim=input_dim,
            hidden_layers=self.nn_config.get("hidden_layers", [256, 128, 64]),
            output_dim=output_dim,
            dropout=self.nn_config.get("dropout", 0.3),
            task_type=self.task_type
        )
        
        # Prepare datasets
        train_dataset = EngagementDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.nn_config.get("batch_size", 32),
            shuffle=True
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_encoded = self.prepare_data(X_val, y_val, fit_scaler=False)
            val_dataset = EngagementDataset(X_val_scaled, y_val_encoded)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Training setup
        if self.task_type == "classification":
            criterion = nn.CrossEntropyLoss() if output_dim > 1 else nn.BCELoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.nn_config.get("learning_rate", 0.001)
        )
        
        # Training loop
        epochs = self.nn_config.get("epochs", 100)
        patience = self.nn_config.get("patience", 10)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                
                if self.task_type == "classification" and output_dim == 1:
                    outputs = outputs.squeeze()
                    batch_targets = batch_targets.float()
                elif self.task_type == "regression":
                    outputs = outputs.squeeze()
                    batch_targets = batch_targets.float()
                
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        outputs = self.model(batch_features)
                        
                        if self.task_type == "classification" and output_dim == 1:
                            outputs = outputs.squeeze()
                            batch_targets = batch_targets.float()
                        elif self.task_type == "regression":
                            outputs = outputs.squeeze()
                            batch_targets = batch_targets.float()
                        
                        loss = criterion(outputs, batch_targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.4f}")
                if val_loader:
                    logger.info(f"Val Loss: {val_loss:.4f}")
        
        self.is_fitted = True
        
        # Generate predictions on training data
        self.model.eval()
        with torch.no_grad():
            train_features_tensor = torch.FloatTensor(X_train)
            predictions_tensor = self.model(train_features_tensor)
            
            if self.task_type == "classification":
                if output_dim > 1:
                    probabilities = predictions_tensor.numpy()
                    predictions = np.argmax(probabilities, axis=1)
                else:
                    probabilities = predictions_tensor.numpy()
                    predictions = (probabilities > 0.5).astype(int).squeeze()
            else:
                predictions = predictions_tensor.squeeze().numpy()
                probabilities = None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, predictions, probabilities)
        
        return ModelResults(
            model=self.model,
            predictions=predictions,
            probabilities=probabilities,
            metrics=metrics,
            feature_importance=None,
            confusion_matrix=confusion_matrix(y_train, predictions) if self.task_type == "classification" else None,
            model_type=self.model_type
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix [N, F]
        
        Returns:
            Predictions [N]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled, _ = self.prepare_data(X, np.zeros(len(X)), fit_scaler=False)
        
        if self.model_type == "neural_network":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                predictions_tensor = self.model(X_tensor)
                
                if self.task_type == "classification":
                    if predictions_tensor.shape[1] > 1:
                        predictions = np.argmax(predictions_tensor.numpy(), axis=1)
                    else:
                        predictions = (predictions_tensor.numpy() > 0.5).astype(int).squeeze()
                else:
                    predictions = predictions_tensor.squeeze().numpy()
        else:
            predictions = self.model.predict(X_scaled)
        
        # Decode predictions for classification
        if self.task_type == "classification" and self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get prediction probabilities for classification tasks.
        
        Args:
            X: Feature matrix [N, F]
        
        Returns:
            Probabilities [N, C] or None for regression
        """
        if self.task_type != "classification":
            return None
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled, _ = self.prepare_data(X, np.zeros(len(X)), fit_scaler=False)
        
        if self.model_type == "neural_network":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                probabilities = self.model(X_tensor).numpy()
        else:
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X_scaled)
            else:
                return None
        
        return probabilities
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        if self.task_type == "classification":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            
            if y_proba is not None:
                from sklearn.metrics import roc_auc_score
                try:
                    if y_proba.shape[1] == 2:  # Binary classification
                        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                    else:  # Multi-class
                        metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                except ValueError:
                    metrics["roc_auc"] = 0.0
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == "neural_network":
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': self.nn_config,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'task_type': self.task_type
            }, filepath)
        else:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type,
                'task_type': self.task_type
            }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if self.model_type == "neural_network":
            checkpoint = torch.load(filepath)
            self.nn_config = checkpoint['model_config']
            self.scaler = checkpoint['scaler']
            self.label_encoder = checkpoint['label_encoder']
            self.task_type = checkpoint['task_type']
            
            # Reconstruct model architecture
            # This requires knowing the input dimension
            logger.warning("Neural network loading requires manual architecture reconstruction")
        else:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.model_type = data['model_type']
            self.task_type = data['task_type']
        
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


def cross_validate_model(
    predictor: EngagementPredictor,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    scoring: str = "f1_weighted"
) -> Dict[str, float]:
    """
    Perform cross-validation on the engagement predictor.
    
    Args:
        predictor: EngagementPredictor instance
        X: Feature matrix
        y: Target values
        cv_folds: Number of CV folds
        scoring: Scoring metric
    
    Returns:
        Cross-validation results
    """
    if predictor.model_type == "neural_network":
        logger.warning("Cross-validation not directly supported for neural networks")
        return {}
    
    # Use the underlying sklearn-compatible model
    X_scaled, y_encoded = predictor.prepare_data(X, y, fit_scaler=True)
    
    scores = cross_val_score(
        predictor.model,
        X_scaled,
        y_encoded,
        cv=cv_folds,
        scoring=scoring
    )
    
    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "all_scores": scores
    }


if __name__ == "__main__":
    # Example usage
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 50)  # 1000 samples, 50 features
    y = np.random.randint(0, 3, 1000)  # 3 engagement classes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_predictor = EngagementPredictor(
        model_type="random_forest",
        task_type="classification"
    )
    
    results = rf_predictor.fit(X_train, y_train)
    print(f"Training accuracy: {results.metrics['accuracy']:.3f}")
    
    # Make predictions
    predictions = rf_predictor.predict(X_test)
    print(f"Test predictions shape: {predictions.shape}")
    
    # Cross-validation
    cv_results = cross_validate_model(rf_predictor, X_train, y_train)
    print(f"CV score: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")