"""
Model Evaluation Module for Predictive Maintenance System

This module handles:
- Comprehensive metrics computation
- Confusion matrix visualization
- ROC curve analysis
- Feature importance (SHAP values)
- Model comparison plots
"""

import os
import sys

# Suppress joblib CPU detection warning on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)
# Disable parallel processing in joblib to avoid the warning
os.environ['JOBLIB_MULTIPROCESSING'] = '0'

import logging
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Configure joblib to use threading instead of multiprocessing
import contextlib

# Suppress stderr temporarily to hide the incomplete traceback
@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output."""
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

# Interpretability
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """

    def __init__(self, data_dir: str = None,
                 model_dir: str = None,
                 output_dir: str = None):
        """
        Initialize the evaluator.

        Args:
            data_dir: Directory containing processed data (default: auto-detect)
            model_dir: Directory containing trained models (default: auto-detect)
            output_dir: Directory to save evaluation outputs (default: auto-detect)
        """
        # Use absolute paths relative to script location
        script_dir = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else script_dir / 'data' / 'processed'
        self.model_dir = Path(model_dir) if model_dir else script_dir / 'models'
        self.output_dir = Path(output_dir) if output_dir else script_dir / 'evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ModelEvaluator initialized")

    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load test data.

        Returns:
            Dictionary with test data
        """
        logger.info("Loading test data...")

        X_test = np.load(self.data_dir / 'X_test.npy')
        y_test = np.load(self.data_dir / 'y_test.npy')

        logger.info(f"Test data loaded: X_test {X_test.shape}, y_test {y_test.shape}")

        return {'X_test': X_test, 'y_test': y_test}

    def load_model(self, model_name: str) -> Any:
        """
        Load a trained model.

        Args:
            model_name: Name of the model file (without .pkl)

        Returns:
            Loaded model
        """
        model_path = self.model_dir / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Suppress stderr when loading LightGBM models to avoid joblib warning
        with suppress_stderr():
            model = joblib.load(model_path)
        logger.info(f"Loaded model: {model_name}")

        return model

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
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
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = 'Confusion Matrix', save_path: str = None):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save the figure
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Failure', 'Failure'],
                   yticklabels=['No Failure', 'Failure'])
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      title: str = 'ROC Curve', save_path: str = None):
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")

        plt.close()

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   title: str = 'Precision-Recall Curve', save_path: str = None):
        """
        Plot Precision-Recall curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")

        plt.close()

    def plot_feature_importance(self, model: Any, feature_names: List[str] = None,
                               title: str = 'Feature Importance', save_path: str = None):
        """
        Plot feature importance for tree-based models.

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the figure
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return

        importances = model.feature_importances_

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]

        # Sort by importance
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], color='skyblue', edgecolor='black')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.close()

    def compute_shap_values(self, model: Any, X: np.ndarray,
                          feature_names: List[str] = None,
                          max_samples: int = 500, save_path: str = None):
        """
        Compute and plot SHAP values for model interpretability.

        Args:
            model: Trained model
            X: Input features
            feature_names: List of feature names
            max_samples: Maximum samples for SHAP computation
            save_path: Path to save the figure
        """
        logger.info("Computing SHAP values...")

        try:
            # Sample data if too large
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X_sample[:100])

            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)

            # Handle multi-output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take positive class

            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")

            plt.close()

            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                            plot_type='bar', show=False)
            plt.tight_layout()

            if save_path:
                bar_path = save_path.replace('.png', '_bar.png')
                plt.savefig(bar_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP bar plot saved to {bar_path}")

            plt.close()

            logger.info("SHAP analysis complete")

        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")

    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.

        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {model_name}...")

        # Load model
        model = self.load_model(model_name)

        # Predictions
        y_pred = model.predict(X_test)

        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = None

        # Metrics
        metrics = self.compute_metrics(y_test, y_pred, y_pred_proba)

        # Classification report
        report = classification_report(y_test, y_pred,
                                      target_names=['No Failure', 'Failure'],
                                      output_dict=True)

        # Create output directory for this model
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Visualizations
        self.plot_confusion_matrix(
            y_test, y_pred,
            title=f'{model_name} - Confusion Matrix',
            save_path=str(model_output_dir / 'confusion_matrix.png')
        )

        if y_pred_proba is not None:
            self.plot_roc_curve(
                y_test, y_pred_proba,
                title=f'{model_name} - ROC Curve',
                save_path=str(model_output_dir / 'roc_curve.png')
            )

            self.plot_precision_recall_curve(
                y_test, y_pred_proba,
                title=f'{model_name} - Precision-Recall Curve',
                save_path=str(model_output_dir / 'pr_curve.png')
            )

        # Feature importance
        self.plot_feature_importance(
            model, feature_names,
            title=f'{model_name} - Feature Importance',
            save_path=str(model_output_dir / 'feature_importance.png')
        )

        # SHAP values
        if model_name in ['xgboost', 'lightgbm', 'random_forest']:
            self.compute_shap_values(
                model, X_test, feature_names,
                save_path=str(model_output_dir / 'shap_summary.png')
            )

        # Save metrics
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': report
        }

        with open(model_output_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Evaluation complete for {model_name}")
        logger.info(f"Metrics: {metrics}")

        return results

    def compare_models(self, model_names: List[str], X_test: np.ndarray,
                      y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models side by side.

        Args:
            model_names: List of model names
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing models...")

        results = []

        for model_name in model_names:
            try:
                model = self.load_model(model_name)
                y_pred = model.predict(X_test)

                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                except:
                    y_pred_proba = None

                metrics = self.compute_metrics(y_test, y_pred, y_pred_proba)
                metrics['model'] = model_name
                results.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")

        # Create comparison DataFrame
        df_results = pd.DataFrame(results)

        # Reorder columns - ensure 'model' is always first
        cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
        available_cols = [col for col in cols if col in df_results.columns]

        # Ensure 'model' column exists
        if 'model' not in df_results.columns and len(df_results) > 0:
            logger.warning("'model' column not found in results, adding from index")
            df_results['model'] = df_results.index

        df_results = df_results[available_cols]

        # Save comparison
        comparison_path = self.output_dir / 'model_comparison.csv'
        df_results.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to {comparison_path}")

        # Plot comparison
        self.plot_model_comparison(df_results, save_path=self.output_dir / 'model_comparison.png')

        return df_results

    def plot_model_comparison(self, df_results: pd.DataFrame, save_path: str = None):
        """
        Plot model comparison chart.

        Args:
            df_results: DataFrame with model comparison results
            save_path: Path to save the figure
        """
        if len(df_results) == 0:
            logger.warning("No results to plot")
            return

        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metrics = [m for m in metrics if m in df_results.columns]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(df_results))
        width = 0.15

        for i, metric in enumerate(metrics):
            offset = width * (i - len(metrics) / 2)
            ax.bar(x + offset, df_results[metric], width, label=metric.replace('_', ' ').title())

        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)

        # Use model names for labels - check if column exists
        if 'model' in df_results.columns:
            ax.set_xticklabels(df_results['model'], rotation=45, ha='right')
        else:
            ax.set_xticklabels(df_results.index, rotation=45, ha='right')

        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.close()


def main():
    """
    Main execution function.
    """
    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Load test data
    data = evaluator.load_data()
    X_test = data['X_test']
    y_test = data['y_test']

    # Feature names
    feature_names = [
        'Air temperature',
        'Process temperature',
        'Rotational speed',
        'Torque',
        'Tool wear',
        'Type'
    ]

    # Models to evaluate
    model_names = [
        'logistic_regression',
        'random_forest',
        'xgboost',
        'lightgbm',
        'xgboost_optimized'
    ]

    # Evaluate each model
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)

    for model_name in model_names:
        try:
            print(f"\nEvaluating {model_name}...")
            evaluator.evaluate_model(model_name, X_test, y_test, feature_names)
        except FileNotFoundError:
            print(f"Model {model_name} not found, skipping...")
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")

    # Compare models
    print("\n" + "="*60)
    print("COMPARING MODELS")
    print("="*60)

    # Use the evaluator's model_dir to check for available models
    available_models = [m for m in model_names if (evaluator.model_dir / f"{m}.pkl").exists()]

    if not available_models:
        print("No models found for comparison.")
    else:
        df_comparison = evaluator.compare_models(available_models, X_test, y_test)
        print("\n" + df_comparison.to_string(index=False))

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {evaluator.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
