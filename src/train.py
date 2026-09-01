"""
Model Training Module for Predictive Maintenance System

This module handles:
- Multiple model training (LogReg, RF, XGBoost, LightGBM, PyTorch NN)
- Hyperparameter optimization with Optuna
- MLflow experiment tracking
- Cross-validation
- Model persistence
"""

import logging
import os
import sys
from contextlib import contextmanager
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
# Suppress joblib CPU detection warnings
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
os.environ['JOBLIB_MULTIPROCESSING'] = '0'


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

# ML Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import xgboost as xgb
import lightgbm as lgb
import optuna

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PyTorchClassifier(nn.Module):
    """
    Feedforward Neural Network for binary classification.
    """

    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout: float = 0.3):
        """
        Initialize the neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(PyTorchClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ModelTrainer:
    """
    Comprehensive model training pipeline with MLflow tracking.
    """

    def __init__(self, data_dir: str = None, model_dir: str = None):
        """
        Initialize the trainer.

        Args:
            data_dir: Directory containing processed data (default: auto-detect)
            model_dir: Directory to save trained models (default: auto-detect)
        """
        # Use absolute paths relative to script location
        script_dir = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else script_dir / 'data' / 'processed'
        self.model_dir = Path(model_dir) if model_dir else script_dir / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        logger.info("ModelTrainer initialized")

    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load preprocessed data.

        Returns:
            Dictionary with train/val/test data
        """
        logger.info("Loading preprocessed data...")

        data = {}
        for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
            filepath = self.data_dir / f"{key}.npy"
            data[key] = np.load(filepath)
            logger.info(f"Loaded {key}: shape {data[key].shape}")

        return data

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, Dict]:
        """
        Train Logistic Regression model.
        """
        logger.info("Training Logistic Regression...")


            # Model
        model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='lbfgs'
            )

            # Train
        model.fit(X_train, y_train)

            # Predict
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Metrics
        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba)

            # Create input example for signature inference
        input_example = X_train[:5]
        

        logger.info(f"Logistic Regression - Metrics: {metrics}")

        return model, metrics

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, Dict]:
        """
        Train Random Forest model.
        """
        logger.info("Training Random Forest...")

        
            # Model
        model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )

            # Train
        model.fit(X_train, y_train)

            # Predict
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Metrics
        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba)

            # Feature importance
        feature_importance = model.feature_importances_

        

        logger.info(f"Random Forest - Metrics: {metrics}")

        return model, metrics

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, Dict]:
        """
        Train XGBoost model.
        """
        logger.info("Training XGBoost...")

       
            # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            # Model
        model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )

            # Train
        model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Predict
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Metrics
        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba)

            

        logger.info(f"XGBoost - Metrics: {metrics}")

        return model, metrics

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, Dict]:
        """
        Train LightGBM model.
        """
        logger.info("Training LightGBM...")

        
            # Model
        model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )

            # Train
        model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )

            # Predict
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Metrics
        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba)

          
        logger.info(f"LightGBM - Metrics: {metrics}")

        return model, metrics

    def train_pytorch_nn(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        epochs: int = 100, batch_size: int = 64) -> Tuple[Any, Dict]:
        """
        Train PyTorch Neural Network.
        """
        logger.info("Training PyTorch Neural Network...")

      
            # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

            # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Model
        input_dim = X_train.shape[1]
        model = PyTorchClassifier(
                input_dim=input_dim,
                hidden_dims=[128, 64, 32],
                dropout=0.3
            ).to(self.device)

            # Loss and optimizer
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)

            # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15

        for epoch in range(epochs):
                model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t).squeeze()
                    val_loss = criterion(val_outputs, y_val_t).item()

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            # Load best model
        model.load_state_dict(best_model_state)

            # Predict
        model.eval()
        with torch.no_grad():
                y_pred_proba = model(X_val_t).squeeze().cpu().numpy()
                y_pred = (y_pred_proba > 0.5).astype(int)

            # Metrics
        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba)

            

            # Save PyTorch model
        model_path = self.model_dir / 'pytorch_model.pt'
        torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'hidden_dims': [128, 64, 32]
            }, model_path)
        

        logger.info(f"PyTorch NN - Metrics: {metrics}")

        return model, metrics

    def optimize_xgboost_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               n_trials: int = 20) -> Tuple[Any, Dict]:
        """
        Optimize XGBoost hyperparameters using Optuna.
        """
        logger.info("Starting Optuna hyperparameter optimization for XGBoost...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_pred_proba)

            return roc_auc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best ROC-AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Train final model with best params
       
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['eval_metric'] = 'logloss'
        best_params['use_label_encoder'] = False

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = self.evaluate_model(y_val, y_pred, y_pred_proba)

        

            

        logger.info(f"XGBoost Optimized - Metrics: {metrics}")

        return model, metrics

    def train_all_models(self) -> Dict[str, Dict]:
        """
        Train all models and compare results.
        """
        logger.info("="*60)
        logger.info("Training all models...")
        logger.info("="*60)

        # Load data
        data = self.load_data()
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']

        results = {}

        # Train models
        models_to_train = [
            ('Logistic Regression', self.train_logistic_regression),
            ('Random Forest', self.train_random_forest),
            ('XGBoost', self.train_xgboost),
            ('LightGBM', self.train_lightgbm),
            ('PyTorch NN', self.train_pytorch_nn),
        ]

        for model_name, train_func in models_to_train:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_name}...")
                logger.info(f"{'='*60}")

                model, metrics = train_func(X_train, y_train, X_val, y_val)
                results[model_name] = {
                    'model': model,
                    'metrics': metrics
                }

                # Save model
                model_path = self.model_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
                with suppress_stderr():
                    joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} to {model_path}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}

        # Optuna optimization
        try:
            logger.info(f"\n{'='*60}")
            logger.info("Running Optuna optimization...")
            logger.info(f"{'='*60}")
            model, metrics = self.optimize_xgboost_optuna(X_train, y_train, X_val, y_val, n_trials=20)
            results['XGBoost Optimized'] = {
                'model': model,
                'metrics': metrics
            }
            model_path = self.model_dir / "xgboost_optimized.pkl"
            with suppress_stderr():
                joblib.dump(model, model_path)
        except Exception as e:
            logger.error(f"Error in Optuna optimization: {str(e)}")

        return results

    def print_results_summary(self, results: Dict[str, Dict]):
        """
        Print a summary table of all model results.
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-"*80)

        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"{model_name:<25} "
                      f"{metrics.get('accuracy', 0):<12.4f} "
                      f"{metrics.get('precision', 0):<12.4f} "
                      f"{metrics.get('recall', 0):<12.4f} "
                      f"{metrics.get('f1_score', 0):<12.4f} "
                      f"{metrics.get('roc_auc', 0):<12.4f}")

        print("="*80)


def main():
    """
    Main execution function.
    """
    # Initialize trainer
    trainer = ModelTrainer()

    # Train all models
    results = trainer.train_all_models()

    # Print summary
    trainer.print_results_summary(results)

    print("\nTraining complete!")
    
    print(f"Models saved to: {trainer.model_dir}")


if __name__ == '__main__':
    main()
