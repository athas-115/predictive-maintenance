"""
Data Preprocessing Module for Predictive Maintenance System

This module handles:
- Data loading and exploration
- Feature encoding (categorical to numerical)
- Feature scaling and normalization
- Class imbalance handling (SMOTE)
- Train/validation/test split
- Data persistence
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for predictive maintenance.
    """

    def __init__(self, data_path: str, output_dir: str = './data/processed'):
        """
        Initialize the preprocessor.

        Args:
            data_path: Path to the raw CSV file
            output_dir: Directory to save processed data
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_col = 'Machine failure'

        logger.info(f"DataPreprocessor initialized with data from {data_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load the raw dataset from CSV.

        Returns:
            DataFrame with raw data
        """
        logger.info("Loading dataset...")
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def explore_data(self, df: pd.DataFrame) -> Dict:
        """
        Perform basic exploratory data analysis.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with data statistics
        """
        logger.info("Performing exploratory data analysis...")

        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'target_distribution': df[self.target_col].value_counts().to_dict(),
            'numeric_summary': df.describe().to_dict()
        }

        # Log key insights
        logger.info(f"Dataset shape: {stats['shape']}")
        logger.info(f"Missing values: {sum(stats['missing_values'].values())} total")
        logger.info(f"Target distribution: {stats['target_distribution']}")

        # Calculate imbalance ratio
        target_counts = df[self.target_col].value_counts()
        imbalance_ratio = target_counts.min() / target_counts.max()
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.4f}")

        return stats

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")

        df_encoded = df.copy()

        # Encode 'Type' column (L, M, H)
        if 'Type' in df_encoded.columns:
            df_encoded['Type_Encoded'] = self.label_encoder.fit_transform(df_encoded['Type'])
            logger.info(f"Type encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # Alternative: One-hot encoding for Type
        type_dummies = pd.get_dummies(df['Type'], prefix='Type', drop_first=False)
        df_encoded = pd.concat([df_encoded, type_dummies], axis=1)

        return df_encoded

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select and prepare features for modeling.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Selecting features...")

        # Features to use for prediction
        feature_cols = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Type_Encoded',  # Using encoded version
            # 'Type_H', 'Type_L', 'Type_M'  # Alternative: one-hot encoded
        ]

        # Verify all columns exist
        available_features = [col for col in feature_cols if col in df.columns]

        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features: {missing}")

        X = df[available_features].copy()
        y = df[self.target_col].copy()

        self.feature_names = available_features
        logger.info(f"Selected {len(available_features)} features: {available_features}")
        logger.info(f"Target: {self.target_col}")

        return X, y

    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                      X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features

        Returns:
            Tuple of scaled arrays (train, val, test)
        """
        logger.info("Scaling features...")

        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Feature scaling complete. Mean: {self.scaler.mean_[:3]}, Std: {self.scaler.scale_[:3]}")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def handle_imbalance(self, X_train: np.ndarray, y_train: np.ndarray,
                        method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE or class weights.

        Args:
            X_train: Training features
            y_train: Training labels
            method: Method to use ('smote' or 'none')

        Returns:
            Tuple of balanced (X_train, y_train)
        """
        logger.info(f"Handling class imbalance using method: {method}")

        original_counts = np.bincount(y_train.astype(int))
        logger.info(f"Original class distribution: {original_counts}")

        if method == 'smote':
            # Apply SMOTE
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            new_counts = np.bincount(y_train_balanced.astype(int))
            logger.info(f"Balanced class distribution: {new_counts}")

            return X_train_balanced, y_train_balanced
        else:
            logger.info("No resampling applied. Consider using class_weight in model training.")
            return X_train, y_train

    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.15, val_size: float = 0.15,
                   random_state: int = 42) -> Dict:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with all splits
        """
        logger.info(f"Splitting data: train={1-test_size-val_size:.2f}, val={val_size:.2f}, test={test_size:.2f}")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train vs val
        val_proportion = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_proportion, random_state=random_state, stratify=y_temp
        )

        logger.info(f"Train size: {X_train.shape[0]}")
        logger.info(f"Validation size: {X_val.shape[0]}")
        logger.info(f"Test size: {X_test.shape[0]}")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

    def save_processed_data(self, data_dict: Dict, metadata: Dict = None):
        """
        Save processed data and artifacts to disk.

        Args:
            data_dict: Dictionary with train/val/test splits
            metadata: Additional metadata to save
        """
        logger.info(f"Saving processed data to {self.output_dir}...")

        # Save datasets
        for key, value in data_dict.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                filepath = self.output_dir / f"{key}.csv"
                value.to_csv(filepath, index=False)
            else:  # numpy array
                filepath = self.output_dir / f"{key}.npy"
                np.save(filepath, value)
            logger.info(f"Saved {key} to {filepath}")

        # Save scaler
        scaler_path = self.output_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        # Save label encoder
        encoder_path = self.output_dir / 'label_encoder.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        logger.info(f"Saved label encoder to {encoder_path}")

        # Save metadata
        if metadata:
            metadata_path = self.output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                # Convert non-serializable objects to native Python types
                metadata_clean = self._make_json_serializable(metadata)
                json.dump(metadata_clean, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

        logger.info("All data saved successfully!")

    def _make_json_serializable(self, obj):
        """
        Recursively convert objects to JSON-serializable types.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif hasattr(obj, 'item'):  # NumPy scalar
            return obj.item()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    def visualize_distributions(self, df: pd.DataFrame, save_path: str = None):
        """
        Create visualizations of data distributions.

        Args:
            df: Input DataFrame
            save_path: Path to save the figure
        """
        logger.info("Creating distribution visualizations...")

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Feature Distributions', fontsize=16)

        # Numeric columns to plot
        numeric_cols = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Machine failure'
        ]

        for idx, col in enumerate(numeric_cols):
            if col in df.columns:
                row = idx // 3
                col_idx = idx % 3

                if col == 'Machine failure':
                    df[col].value_counts().plot(kind='bar', ax=axes[row, col_idx])
                else:
                    df[col].hist(bins=50, ax=axes[row, col_idx], edgecolor='black')

                axes[row, col_idx].set_title(col)
                axes[row, col_idx].set_xlabel('')

        # Remove empty subplots
        for idx in range(len(numeric_cols), 9):
            row = idx // 3
            col_idx = idx % 3
            fig.delaxes(axes[row, col_idx])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")

        plt.close()

    def process_pipeline(self, use_smote: bool = True, save_visualizations: bool = True) -> Dict:
        """
        Execute the complete preprocessing pipeline.

        Args:
            use_smote: Whether to apply SMOTE for class imbalance
            save_visualizations: Whether to save distribution plots

        Returns:
            Dictionary with processed data
        """
        logger.info("="*60)
        logger.info("Starting preprocessing pipeline...")
        logger.info("="*60)

        # Step 1: Load data
        df = self.load_data()

        # Step 2: Explore data
        stats = self.explore_data(df)

        # Step 3: Visualize distributions
        if save_visualizations:
            viz_path = self.output_dir / 'distributions.png'
            self.visualize_distributions(df, str(viz_path))

        # Step 4: Encode features
        df_encoded = self.encode_features(df)

        # Step 5: Select features
        X, y = self.select_features(df_encoded)

        # Step 6: Split data
        splits = self.split_data(X, y)

        # Step 7: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            splits['X_train'], splits['X_val'], splits['X_test']
        )

        # Step 8: Handle imbalance (only on training set)
        if use_smote:
            X_train_balanced, y_train_balanced = self.handle_imbalance(
                X_train_scaled, splits['y_train'].values
            )
        else:
            X_train_balanced = X_train_scaled
            y_train_balanced = splits['y_train'].values

        # Prepare final data dictionary
        processed_data = {
            'X_train': X_train_balanced,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_balanced,
            'y_val': splits['y_val'].values,
            'y_test': splits['y_test'].values,
        }

        # Metadata
        metadata = {
            'feature_names': self.feature_names,
            'target_name': self.target_col,
            'train_size': len(X_train_balanced),
            'val_size': len(X_val_scaled),
            'test_size': len(X_test_scaled),
            'smote_applied': use_smote,
            'statistics': stats
        }

        # Step 9: Save everything
        self.save_processed_data(processed_data, metadata)

        logger.info("="*60)
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info("="*60)

        return processed_data


def main():
    """
    Main execution function.
    """
    # Configuration - Use absolute paths relative to this script
    SCRIPT_DIR = Path(__file__).parent.parent  # Go up to project root
    DATA_PATH = SCRIPT_DIR / 'data' / 'raw' / 'ai4i2020.csv'
    OUTPUT_DIR = SCRIPT_DIR / 'data' / 'processed'

    # Initialize preprocessor
    preprocessor = DataPreprocessor(str(DATA_PATH), str(OUTPUT_DIR))

    # Run pipeline
    processed_data = preprocessor.process_pipeline(
        use_smote=True,
        save_visualizations=True
    )

    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Train samples: {processed_data['X_train'].shape[0]}")
    print(f"Validation samples: {processed_data['X_val'].shape[0]}")
    print(f"Test samples: {processed_data['X_test'].shape[0]}")
    print(f"Number of features: {processed_data['X_train'].shape[1]}")
    print(f"Feature names: {preprocessor.feature_names}")
    print("="*60)
    print("Data preprocessing complete!")
    print(f"Processed data saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
