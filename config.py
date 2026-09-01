"""
Configuration Module for Predictive Maintenance System

Centralized configuration management using environment variables and defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Application configuration class.
    """

    # Base paths
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODEL_DIR = ROOT_DIR / 'models'
    EVALUATION_DIR = ROOT_DIR / 'evaluation'
    LOGS_DIR = ROOT_DIR / 'logs'
    MONITORING_LOGS_DIR = ROOT_DIR / 'monitoring_logs'
    RETRAINING_LOGS_DIR = ROOT_DIR / 'retraining_logs'

    # Data paths
    DATASET_PATH = RAW_DATA_DIR / 'ai4i2020.csv'
    SCALER_PATH = PROCESSED_DATA_DIR / 'scaler.pkl'
    LABEL_ENCODER_PATH = PROCESSED_DATA_DIR / 'label_encoder.pkl'
    METADATA_PATH = PROCESSED_DATA_DIR / 'metadata.json'

    # Model paths
    DEFAULT_MODEL_NAME = os.getenv('MODEL_NAME', 'lightgbm')
    MODEL_PATH = MODEL_DIR / f'{DEFAULT_MODEL_NAME}.pkl'
    PYTORCH_MODEL_PATH = MODEL_DIR / 'pytorch_model.pt'

    # MLflow configuration
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', './mlflow_logs')
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'predictive_maintenance')

    # API configuration
    API_HOST = os.getenv('API_HOST', 'localhost')
    API_PORT = int(os.getenv('API_PORT', 8000))
    API_RELOAD = os.getenv('API_RELOAD', 'True').lower() == 'true'
    API_WORKERS = int(os.getenv('API_WORKERS', 1))

    # Dashboard configuration
    DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', 8501))
    DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', 'localhost')

    # Training configuration
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
    TEST_SIZE = float(os.getenv('TEST_SIZE', 0.15))
    VAL_SIZE = float(os.getenv('VAL_SIZE', 0.15))
    USE_SMOTE = os.getenv('USE_SMOTE', 'True').lower() == 'true'

    # Model hyperparameters
    LGBM_N_ESTIMATORS = int(os.getenv('LGBM_N_ESTIMATORS', 200))
    LGBM_MAX_DEPTH = int(os.getenv('LGBM_MAX_DEPTH', 6))
    LGBM_LEARNING_RATE = float(os.getenv('LGBM_LEARNING_RATE', 0.1))

    XGB_N_ESTIMATORS = int(os.getenv('XGB_N_ESTIMATORS', 200))
    XGB_MAX_DEPTH = int(os.getenv('XGB_MAX_DEPTH', 6))
    XGB_LEARNING_RATE = float(os.getenv('XGB_LEARNING_RATE', 0.1))

    # Optuna configuration
    OPTUNA_N_TRIALS = int(os.getenv('OPTUNA_N_TRIALS', 20))
    OPTUNA_TIMEOUT = Optional[int](os.getenv('OPTUNA_TIMEOUT', None))

    # Monitoring configuration
    DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD', 0.05))
    PERFORMANCE_THRESHOLD = float(os.getenv('PERFORMANCE_THRESHOLD', 0.05))
    MONITORING_SAMPLE_SIZE = int(os.getenv('MONITORING_SAMPLE_SIZE', 500))

    # Retraining configuration
    RETRAIN_THRESHOLD = float(os.getenv('RETRAIN_THRESHOLD', 0.3))
    RETRAIN_MIN_SAMPLES = int(os.getenv('RETRAIN_MIN_SAMPLES', 1000))
    RETRAIN_FORCE = os.getenv('RETRAIN_FORCE', 'False').lower() == 'true'

    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv(
        'LOG_FORMAT',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Feature names
    FEATURE_NAMES = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Type_Encoded'
    ]

    FEATURE_NAMES_DISPLAY = [
        'Air Temperature',
        'Process Temperature',
        'Rotational Speed',
        'Torque',
        'Tool Wear',
        'Type'
    ]

    # Target name
    TARGET_NAME = 'Machine failure'

    # Product types
    PRODUCT_TYPES = ['L', 'M', 'H']

    # Failure types
    FAILURE_TYPES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    @classmethod
    def create_directories(cls):
        """
        Create necessary directories if they don't exist.
        """
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.EVALUATION_DIR,
            cls.LOGS_DIR,
            cls.MONITORING_LOGS_DIR,
            cls.RETRAINING_LOGS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid
        """
        errors = []

        # Check if dataset exists
        if not cls.DATASET_PATH.exists():
            errors.append(f"Dataset not found at {cls.DATASET_PATH}")

        # Validate thresholds
        if not (0 <= cls.DRIFT_THRESHOLD <= 1):
            errors.append("DRIFT_THRESHOLD must be between 0 and 1")

        if not (0 <= cls.RETRAIN_THRESHOLD <= 1):
            errors.append("RETRAIN_THRESHOLD must be between 0 and 1")

        if not (0 <= cls.TEST_SIZE <= 1):
            errors.append("TEST_SIZE must be between 0 and 1")

        if not (0 <= cls.VAL_SIZE <= 1):
            errors.append("VAL_SIZE must be between 0 and 1")

        # Print errors if any
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    @classmethod
    def print_config(cls):
        """
        Print current configuration.
        """
        print("\n" + "="*60)
        print("CONFIGURATION SETTINGS")
        print("="*60)

        print("\n[Paths]")
        print(f"  Root Directory: {cls.ROOT_DIR}")
        print(f"  Dataset: {cls.DATASET_PATH}")
        print(f"  Model Directory: {cls.MODEL_DIR}")

        print("\n[API]")
        print(f"  Host: {cls.API_HOST}")
        print(f"  Port: {cls.API_PORT}")
        print(f"  Reload: {cls.API_RELOAD}")

        print("\n[Training]")
        print(f"  Random State: {cls.RANDOM_STATE}")
        print(f"  Test Size: {cls.TEST_SIZE}")
        print(f"  Validation Size: {cls.VAL_SIZE}")
        print(f"  Use SMOTE: {cls.USE_SMOTE}")

        print("\n[Monitoring]")
        print(f"  Drift Threshold: {cls.DRIFT_THRESHOLD}")
        print(f"  Performance Threshold: {cls.PERFORMANCE_THRESHOLD}")

        print("\n[Retraining]")
        print(f"  Retrain Threshold: {cls.RETRAIN_THRESHOLD}")
        print(f"  Min Samples: {cls.RETRAIN_MIN_SAMPLES}")

        print("\n[MLflow]")
        print(f"  Tracking URI: {cls.MLFLOW_TRACKING_URI}")
        print(f"  Experiment: {cls.MLFLOW_EXPERIMENT_NAME}")

        print("\n" + "="*60 + "\n")


# Create directories on import
Config.create_directories()


if __name__ == '__main__':
    # Print and validate configuration
    Config.print_config()

    if Config.validate_config():
        print("Configuration is valid")
    else:
        print("Configuration has errors")
