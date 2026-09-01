"""
Automated Retraining Module for Predictive Maintenance System

This module handles:
- Automated model retraining based on drift detection
- Data collection and preparation
- Model evaluation and comparison
- Model deployment decision
"""

import logging
import os
import sys
from contextlib import contextmanager
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
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


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb


from monitor import DataDriftDetector, ModelPerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomatedRetrainer:
    """
    Automated model retraining system.
    """

    def __init__(self, data_dir: str = None,
                 model_dir: str = None,
                 retrain_threshold: float = 0.3):
        """
        Initialize retrainer.

        Args:
            data_dir: Directory containing processed data (default: auto-detect)
            model_dir: Directory containing models (default: auto-detect)
            retrain_threshold: Drift threshold to trigger retraining
        """
        # Use absolute paths relative to script location
        script_dir = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else script_dir / 'data' / 'processed'
        self.model_dir = Path(model_dir) if model_dir else script_dir / 'models'
        self.retrain_threshold = retrain_threshold

        self.drift_detector = DataDriftDetector()
        self.performance_monitor = ModelPerformanceMonitor()

        

        logger.info("AutomatedRetrainer initialized")

    def check_retrain_trigger(self, new_data: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check if retraining should be triggered.

        Args:
            new_data: New data to evaluate

        Returns:
            Tuple of (should_retrain, drift_results)
        """
        logger.info("Checking retraining triggers...")

        # Check data drift
        drift_results = self.drift_detector.detect_drift(new_data)
        drift_score = len(drift_results['drifted_features']) / len(drift_results['features'])

        should_retrain = drift_score >= self.retrain_threshold

        logger.info(f"Drift score: {drift_score:.4f} (threshold: {self.retrain_threshold})")
        logger.info(f"Should retrain: {should_retrain}")

        return should_retrain, drift_results

    def prepare_retraining_data(self, new_data: np.ndarray,
                               new_labels: np.ndarray) -> Dict:
        """
        Prepare data for retraining by combining old and new data.

        Args:
            new_data: New feature data
            new_labels: New labels

        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("Preparing retraining data...")

        # Load original training data
        X_train_old = np.load(self.data_dir / 'X_train.npy')
        y_train_old = np.load(self.data_dir / 'y_train.npy')

        # Combine old and new data
        X_combined = np.vstack([X_train_old, new_data])
        y_combined = np.concatenate([y_train_old, new_labels])

        logger.info(f"Combined data size: {X_combined.shape[0]} samples")

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_combined, y_combined, test_size=0.3, random_state=42, stratify=y_combined
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

    def retrain_model(self, data: Dict) -> Tuple[any, Dict]:
        """
        Retrain the model with new data.

        Args:
            data: Dictionary with train/val/test data

        Returns:
            Tuple of (new_model, metrics)
        """
        logger.info("Retraining LightGBM model...")

        
            # Train LightGBM model (champion model)
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

        model.fit(
            data['X_train'], data['y_train'],
            eval_set=[(data['X_val'], data['y_val'])],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        # Evaluate
        y_pred = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]

        metrics = {
            'accuracy': accuracy_score(data['y_test'], y_pred),
            'f1_score': f1_score(data['y_test'], y_pred),
            'roc_auc': roc_auc_score(data['y_test'], y_pred_proba),
            'train_samples': len(data['X_train']),
            'retrain_timestamp': datetime.now().isoformat()
        }

       

        logger.info(f"Retraining complete - Metrics: {metrics}")

        return model, metrics

    def compare_models(self, new_model: any, new_metrics: Dict,
                      old_model_path: str) -> bool:
        """
        Compare new model with old model.

        Args:
            new_model: Newly trained model
            new_metrics: New model metrics
            old_model_path: Path to old model

        Returns:
            True if new model is better
        """
        logger.info("Comparing models...")

        try:
            # Load old model
            with suppress_stderr():
                old_model = joblib.load(old_model_path)

            # Load test data
            X_test = np.load(self.data_dir / 'X_test.npy')
            y_test = np.load(self.data_dir / 'y_test.npy')

            # Evaluate old model
            y_pred_old = old_model.predict(X_test)
            y_pred_proba_old = old_model.predict_proba(X_test)[:, 1]

            old_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_old),
                'f1_score': f1_score(y_test, y_pred_old),
                'roc_auc': roc_auc_score(y_test, y_pred_proba_old)
            }

            # Compare
            improvement = {
                'accuracy': new_metrics['accuracy'] - old_metrics['accuracy'],
                'f1_score': new_metrics['f1_score'] - old_metrics['f1_score'],
                'roc_auc': new_metrics['roc_auc'] - old_metrics['roc_auc']
            }

            logger.info(f"Old model - Accuracy: {old_metrics['accuracy']:.4f}, F1: {old_metrics['f1_score']:.4f}")
            logger.info(f"New model - Accuracy: {new_metrics['accuracy']:.4f}, F1: {new_metrics['f1_score']:.4f}")
            logger.info(f"Improvement: {improvement}")

            # Decision: new model is better if F1 score improved
            is_better = improvement['f1_score'] > 0 or improvement['roc_auc'] > 0.01

            logger.info(f"New model is {'BETTER' if is_better else 'NOT BETTER'}")

            return is_better

        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return False

    def deploy_model(self, model: any, model_name: str = 'lightgbm_retrained'):
        """
        Deploy the new model.

        Args:
            model: Model to deploy
            model_name: Name for the deployed model
        """
        logger.info(f"Deploying model: {model_name}")

        # Save new model
        model_path = self.model_dir / f"{model_name}.pkl"
        with suppress_stderr():
            joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Create backup of old model
        old_model_path = self.model_dir / 'lightgbm.pkl'
        if old_model_path.exists():
            backup_path = self.model_dir / f"lightgbm_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with suppress_stderr():
                old_model_backup = joblib.load(old_model_path)
                joblib.dump(old_model_backup, backup_path)
            logger.info(f"Old model backed up to {backup_path}")

        # Replace with new model
        with suppress_stderr():
            joblib.dump(model, old_model_path)
        logger.info("New model deployed successfully")

    def retrain_pipeline(self, new_data: np.ndarray,
                        new_labels: np.ndarray,
                        force_retrain: bool = False) -> Dict:
        """
        Execute the complete retraining pipeline.

        Args:
            new_data: New feature data
            new_labels: New labels
            force_retrain: Force retraining regardless of drift

        Returns:
            Retraining report
        """
        logger.info("="*60)
        logger.info("Starting retraining pipeline...")
        logger.info("="*60)

        report = {
            'timestamp': datetime.now().isoformat(),
            'retrain_triggered': False,
            'model_deployed': False
        }

        # Step 1: Check if retraining is needed
        if not force_retrain:
            should_retrain, drift_results = self.check_retrain_trigger(new_data)
            report['drift_results'] = drift_results

            if not should_retrain:
                logger.info("Retraining not needed - drift below threshold")
                report['message'] = 'Retraining not needed'
                return report

        report['retrain_triggered'] = True

        # Step 2: Prepare data
        data = self.prepare_retraining_data(new_data, new_labels)

        # Step 3: Retrain model
        new_model, new_metrics = self.retrain_model(data)
        report['new_metrics'] = new_metrics

        # Step 4: Compare with old model
        old_model_path = self.model_dir / 'lightgbm.pkl'

        if old_model_path.exists():
            is_better = self.compare_models(new_model, new_metrics, str(old_model_path))
            report['is_better'] = is_better

            # Step 5: Deploy if better
            if is_better:
                self.deploy_model(new_model)
                report['model_deployed'] = True
                report['message'] = 'New model deployed successfully'
            else:
                report['message'] = 'New model not deployed - performance not improved'
        else:
            # No old model, deploy new one
            self.deploy_model(new_model)
            report['model_deployed'] = True
            report['message'] = 'First model deployed'

        logger.info("="*60)
        logger.info("Retraining pipeline complete")
        logger.info("="*60)

        return report


def main():
    """
    Main retraining execution.
    """
    logger.info("Starting automated retraining system...")

    # Initialize retrainer
    retrainer = AutomatedRetrainer()

    # Use absolute paths
    script_dir = Path(__file__).parent.parent

    # Load test data (simulating new production data with labels)
    X_test = np.load(script_dir / 'data' / 'processed' / 'X_test.npy')
    y_test = np.load(script_dir / 'data' / 'processed' / 'y_test.npy')

    # Simulate: Use a subset as "new labeled data"
    sample_size = 1000
    indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_new = X_test[indices]
    y_new = y_test[indices]

    # Optional: Add noise to simulate drift
    # X_new = X_new + np.random.normal(0, 0.2, X_new.shape)

    # Run retraining pipeline
    report = retrainer.retrain_pipeline(X_new, y_new, force_retrain=False)

    # Save report
    report_dir = script_dir / 'retraining_logs'
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / f"retrain_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {report_file}")

    # Print summary
    print("\n" + "="*60)
    print("RETRAINING REPORT SUMMARY")
    print("="*60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Retrain Triggered: {'✅ Yes' if report['retrain_triggered'] else '❌ No'}")

    if report['retrain_triggered']:
        print(f"\nNew Model Metrics:")
        if 'new_metrics' in report:
            metrics = report['new_metrics']
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        print(f"\nModel Deployed: {'✅ Yes' if report['model_deployed'] else '❌ No'}")

    print(f"\nMessage: {report.get('message', 'N/A')}")
    print("="*60)


if __name__ == '__main__':
    main()
