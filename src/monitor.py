"""
Monitoring Module for Predictive Maintenance System

This module handles:
- Data drift detection
- Model performance monitoring
- Alert generation
- Logging and metrics collection
"""

import os
import logging
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import warnings
import sys
import contextlib

# Suppress warnings and joblib issues
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)
os.environ['JOBLIB_MULTIPROCESSING'] = '0'
warnings.filterwarnings('ignore')

# Context manager to suppress stderr
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

from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detect data drift using statistical tests.
    """

    def __init__(self, reference_data_path: str = None,
                 drift_threshold: float = 0.05):
        """
        Initialize drift detector.

        Args:
            reference_data_path: Path to reference (training) data (default: auto-detect)
            drift_threshold: P-value threshold for drift detection
        """
        # Use absolute path relative to script location
        if reference_data_path is None:
            script_dir = Path(__file__).parent.parent
            reference_data_path = str(script_dir / 'data' / 'processed' / 'X_train.npy')

        self.reference_data = np.load(reference_data_path)
        self.drift_threshold = drift_threshold
        self.feature_stats = self._compute_feature_stats(self.reference_data)

        logger.info(f"DataDriftDetector initialized with {self.reference_data.shape[0]} reference samples")

    def _compute_feature_stats(self, data: np.ndarray) -> Dict:
        """
        Compute statistical properties of features.

        Args:
            data: Feature array

        Returns:
            Dictionary with feature statistics
        """
        stats_dict = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0)
        }
        return stats_dict

    def detect_drift(self, new_data: np.ndarray) -> Dict:
        """
        Detect drift using Kolmogorov-Smirnov test.

        Args:
            new_data: New data to compare with reference

        Returns:
            Dictionary with drift results
        """
        logger.info("Detecting data drift...")

        n_features = self.reference_data.shape[1]
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(new_data),
            'features': [],
            'drift_detected': False,
            'drifted_features': []
        }

        for i in range(n_features):
            # Kolmogorov-Smirnov test
            statistic, pvalue = stats.ks_2samp(
                self.reference_data[:, i],
                new_data[:, i]
            )

            is_drifted = pvalue < self.drift_threshold

            feature_result = {
                'feature_index': i,
                'ks_statistic': float(statistic),
                'p_value': float(pvalue),
                'drift_detected': is_drifted,
                'reference_mean': float(self.feature_stats['mean'][i]),
                'new_mean': float(np.mean(new_data[:, i])),
                'reference_std': float(self.feature_stats['std'][i]),
                'new_std': float(np.std(new_data[:, i]))
            }

            drift_results['features'].append(feature_result)

            if is_drifted:
                drift_results['drifted_features'].append(i)
                drift_results['drift_detected'] = True
                logger.warning(f"Drift detected in feature {i}: p-value={pvalue:.4f}")

        if drift_results['drift_detected']:
            logger.warning(f"Data drift detected in {len(drift_results['drifted_features'])} features")
        else:
            logger.info("No significant data drift detected")

        return drift_results

    def compute_drift_score(self, new_data: np.ndarray) -> float:
        """
        Compute overall drift score (0-1).

        Args:
            new_data: New data to evaluate

        Returns:
            Drift score (higher = more drift)
        """
        drift_results = self.detect_drift(new_data)
        drift_score = len(drift_results['drifted_features']) / len(drift_results['features'])
        return drift_score


class ModelPerformanceMonitor:
    """
    Monitor model performance over time.
    """

    def __init__(self, log_dir: str = None):
        """
        Initialize performance monitor.

        Args:
            log_dir: Directory to store monitoring logs (default: auto-detect)
        """
        # Use absolute path relative to script location
        if log_dir is None:
            script_dir = Path(__file__).parent.parent
            log_dir = str(script_dir / 'monitoring_logs')

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.performance_history = []
        self._load_history()

        logger.info("ModelPerformanceMonitor initialized")

    def _load_history(self):
        """
        Load performance history from disk.
        """
        history_file = self.log_dir / 'performance_history.json'

        if history_file.exists():
            with open(history_file, 'r') as f:
                self.performance_history = json.load(f)
            logger.info(f"Loaded {len(self.performance_history)} historical records")

    def _save_history(self):
        """
        Save performance history to disk.
        """
        history_file = self.log_dir / 'performance_history.json'

        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)

        logger.info(f"Saved performance history to {history_file}")

    def log_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = 'default', metadata: Dict = None):
        """
        Log model performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            metadata: Additional metadata to log
        """
        # Compute metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'num_samples': len(y_true),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
        }

        if metadata:
            metrics['metadata'] = metadata

        self.performance_history.append(metrics)
        self._save_history()

        logger.info(f"Logged performance: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")

    def check_performance_degradation(self, threshold: float = 0.05) -> Dict:
        """
        Check for performance degradation.

        Args:
            threshold: Threshold for degradation detection

        Returns:
            Dictionary with degradation results
        """
        if len(self.performance_history) < 2:
            return {'degradation_detected': False, 'message': 'Insufficient history'}

        # Compare recent performance with baseline
        baseline_metrics = self.performance_history[0]
        recent_metrics = self.performance_history[-1]

        degradation_results = {
            'timestamp': datetime.now().isoformat(),
            'degradation_detected': False,
            'degraded_metrics': []
        }

        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            baseline_value = baseline_metrics.get(metric, 0)
            recent_value = recent_metrics.get(metric, 0)

            degradation = baseline_value - recent_value

            if degradation > threshold:
                degradation_results['degradation_detected'] = True
                degradation_results['degraded_metrics'].append({
                    'metric': metric,
                    'baseline': baseline_value,
                    'recent': recent_value,
                    'degradation': degradation
                })

                logger.warning(f"Performance degradation detected in {metric}: "
                             f"{baseline_value:.4f} -> {recent_value:.4f}")

        return degradation_results


class MonitoringSystem:
    """
    Integrated monitoring system.
    """

    def __init__(self):
        """
        Initialize monitoring system.
        """
        self.drift_detector = DataDriftDetector()
        self.performance_monitor = ModelPerformanceMonitor()
        self.alerts = []

        logger.info("MonitoringSystem initialized")

    def monitor(self, new_data: np.ndarray, y_true: np.ndarray = None,
               y_pred: np.ndarray = None, model_name: str = 'default') -> Dict:
        """
        Perform comprehensive monitoring.

        Args:
            new_data: New feature data
            y_true: True labels (optional)
            y_pred: Predicted labels (optional)
            model_name: Name of the model

        Returns:
            Monitoring report
        """
        logger.info("="*60)
        logger.info("Starting monitoring cycle...")
        logger.info("="*60)

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'num_samples': len(new_data)
        }

        # 1. Data drift detection
        drift_results = self.drift_detector.detect_drift(new_data)
        report['drift'] = drift_results

        if drift_results['drift_detected']:
            self.generate_alert('data_drift',
                              f"Data drift detected in {len(drift_results['drifted_features'])} features")

        # 2. Performance monitoring (if labels available)
        if y_true is not None and y_pred is not None:
            self.performance_monitor.log_performance(y_true, y_pred, model_name)

            degradation_results = self.performance_monitor.check_performance_degradation()
            report['performance_degradation'] = degradation_results

            if degradation_results.get('degradation_detected', False):
                self.generate_alert('performance_degradation',
                                  f"Performance degradation detected: {degradation_results}")

        # 3. Alert summary
        report['alerts'] = self.alerts

        logger.info("Monitoring cycle complete")
        return report

    def generate_alert(self, alert_type: str, message: str):
        """
        Generate monitoring alert.

        Args:
            alert_type: Type of alert
            message: Alert message
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': 'high' if 'drift' in alert_type else 'medium'
        }

        self.alerts.append(alert)
        logger.warning(f"ALERT [{alert_type}]: {message}")

    def _make_json_serializable(self, obj):
        """
        Recursively convert non-serializable types to JSON-compatible types.
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    def save_report(self, report: Dict, output_dir: str = '../monitoring_logs'):
        """
        Save monitoring report to disk.

        Args:
            report: Monitoring report dictionary
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f'monitoring_report_{timestamp}.json'

        # Convert report to JSON-serializable format
        serializable_report = self._make_json_serializable(report)

        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)

        logger.info(f"Report saved to {report_file}")


def main():
    """
    Main monitoring execution.
    """
    logger.info("Starting monitoring system...")

    # Initialize monitoring system
    monitor = MonitoringSystem()

    # Use absolute paths
    script_dir = Path(__file__).parent.parent

    # Load test data (simulating new production data)
    X_test = np.load(script_dir / 'data' / 'processed' / 'X_test.npy')
    y_test = np.load(script_dir / 'data' / 'processed' / 'y_test.npy')

    # Simulate: Use a subset as "new data"
    sample_size = 500
    indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_new = X_test[indices]
    y_new = y_test[indices]

    # For demonstration: add some noise to simulate drift
    # X_new = X_new + np.random.normal(0, 0.1, X_new.shape)

    # Load model and make predictions
    import joblib
    model_path = script_dir / 'models' / 'lightgbm.pkl'

    if model_path.exists():
        # Suppress stderr during model loading
        with suppress_stderr():
            model = joblib.load(model_path)
        y_pred = model.predict(X_new)

        # Run monitoring
        report = monitor.monitor(X_new, y_new, y_pred, model_name='lightgbm')

        # Save report
        monitor.save_report(report)

        # Print summary
        print("\n" + "="*60)
        print("MONITORING REPORT SUMMARY")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Model: {report['model_name']}")
        print(f"Samples: {report['num_samples']}")
        print(f"\nData Drift: {'⚠️ DETECTED' if report['drift']['drift_detected'] else '✅ None'}")
        if report['drift']['drift_detected']:
            print(f"Drifted Features: {len(report['drift']['drifted_features'])}")

        if 'performance_degradation' in report:
            print(f"\nPerformance Degradation: {'⚠️ DETECTED' if report['performance_degradation'].get('degradation_detected', False) else '✅ None'}")

        print(f"\nAlerts: {len(report['alerts'])}")
        for alert in report['alerts']:
            print(f"  - [{alert['type']}] {alert['message']}")

        print("="*60)

    else:
        logger.error("Model not found. Train a model first.")


if __name__ == '__main__':
    main()
