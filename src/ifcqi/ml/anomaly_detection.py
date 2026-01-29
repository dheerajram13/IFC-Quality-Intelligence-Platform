"""Anomaly detection using Isolation Forest and other ML models.

Detects geometric anomalies in IFC elements using unsupervised learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ifcqi.logger import get_logger
from ifcqi.ml.preprocessing import preprocess_features

logger = get_logger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a single element."""

    global_id: str
    ifc_type: str
    is_anomaly: bool
    anomaly_score: float  # Lower = more anomalous (negative values)
    anomaly_probability: float  # 0-1, higher = more anomalous


class AnomalyDetector:
    """Isolation Forest-based anomaly detector for IFC geometry."""

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (0.0-0.5)
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw to train each tree
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model: Optional[IsolationForest] = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(
        self, features_df: pd.DataFrame, missing_strategy: str = "median", scaler_type: str = "robust"
    ) -> AnomalyDetector:
        """Train the anomaly detector on features.

        Args:
            features_df: DataFrame with geometry features
            missing_strategy: Strategy for handling missing values
            scaler_type: Type of scaler to use

        Returns:
            self
        """
        # Preprocess features
        X, indices, scaler = preprocess_features(
            features_df, missing_strategy=missing_strategy, scaler_type=scaler_type
        )

        if X.empty:
            raise ValueError("No valid features after preprocessing")

        self.scaler = scaler
        self.feature_names = list(X.columns)

        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X.values)
        self.is_fitted = True

        logger.info(
            f"Trained Isolation Forest on {len(X)} samples, "
            f"{len(self.feature_names)} features, "
            f"contamination={self.contamination}"
        )

        return self

    def predict(self, features_df: pd.DataFrame) -> list[AnomalyResult]:
        """Detect anomalies in new data.

        Args:
            features_df: DataFrame with geometry features

        Returns:
            List of AnomalyResult objects
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Preprocess features using fitted scaler
        X, indices, _ = preprocess_features(
            features_df, scaler=self.scaler, scaler_type="robust"
        )

        if X.empty:
            logger.warning("No valid features to predict")
            return []

        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X.values)

        # Get anomaly scores (lower = more anomalous, typically negative for anomalies)
        scores = self.model.score_samples(X.values)

        # Convert scores to probabilities (0-1, higher = more anomalous)
        # Use sigmoid-like transformation
        probabilities = self._scores_to_probabilities(scores)

        # Build results
        results = []
        for idx, pred, score, prob in zip(indices, predictions, scores, probabilities):
            if idx in features_df.index:
                results.append(
                    AnomalyResult(
                        global_id=str(features_df.loc[idx, "global_id"]),
                        ifc_type=str(features_df.loc[idx, "ifc_type"]),
                        is_anomaly=(pred == -1),
                        anomaly_score=float(score),
                        anomaly_probability=float(prob),
                    )
                )

        n_anomalies = sum(1 for r in results if r.is_anomaly)
        logger.info(
            f"Detected {n_anomalies} anomalies out of {len(results)} elements "
            f"({n_anomalies / max(1, len(results)) * 100:.1f}%)"
        )

        return results

    def _scores_to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """Convert anomaly scores to probabilities.

        Args:
            scores: Raw anomaly scores from Isolation Forest

        Returns:
            Array of probabilities (0-1, higher = more anomalous)
        """
        # Normalize scores to [0, 1] range
        # Lower scores = more anomalous, so we invert
        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.full_like(scores, 0.5)

        # Invert so higher probability = more anomalous
        probabilities = 1.0 - (scores - min_score) / (max_score - min_score)

        return probabilities

    def get_anomalies(
        self, features_df: pd.DataFrame, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Get DataFrame of detected anomalies.

        Args:
            features_df: DataFrame with geometry features
            threshold: Probability threshold (0-1). If None, uses model's contamination

        Returns:
            DataFrame with anomaly information
        """
        results = self.predict(features_df)

        if not results:
            return pd.DataFrame(
                columns=[
                    "global_id",
                    "ifc_type",
                    "is_anomaly",
                    "anomaly_score",
                    "anomaly_probability",
                ]
            )

        df = pd.DataFrame([vars(r) for r in results])

        # Filter by threshold if provided
        if threshold is not None:
            df = df[df["anomaly_probability"] >= threshold].copy()

        return df

    def save_model(self, output_path: Path) -> None:
        """Save the trained model to disk.

        Args:
            output_path: Path to save the model (should end with .pkl)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        import joblib

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, output_path)

        logger.info(f"Saved model to {output_path}")

    @classmethod
    def load_model(cls, model_path: Path) -> AnomalyDetector:
        """Load a trained model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded AnomalyDetector instance
        """
        import joblib

        model_data = joblib.load(model_path)

        detector = cls(
            contamination=model_data["contamination"],
            n_estimators=model_data["n_estimators"],
        )

        detector.model = model_data["model"]
        detector.scaler = model_data["scaler"]
        detector.feature_names = model_data["feature_names"]
        detector.is_fitted = True

        logger.info(f"Loaded model from {model_path}")

        return detector


def detect_anomalies(
    features_df: pd.DataFrame,
    contamination: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[list[AnomalyResult], AnomalyDetector]:
    """Convenience function to train and predict in one call.

    Args:
        features_df: DataFrame with geometry features
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees in the forest
        random_state: Random seed

    Returns:
        Tuple of (anomaly_results, trained_detector)
    """
    detector = AnomalyDetector(
        contamination=contamination, n_estimators=n_estimators, random_state=random_state
    )

    detector.fit(features_df)
    results = detector.predict(features_df)

    return results, detector


def anomalies_to_dataframe(results: list[AnomalyResult]) -> pd.DataFrame:
    """Convert anomaly results to DataFrame.

    Args:
        results: List of AnomalyResult objects

    Returns:
        DataFrame with anomaly information
    """
    if not results:
        return pd.DataFrame(
            columns=[
                "global_id",
                "ifc_type",
                "is_anomaly",
                "anomaly_score",
                "anomaly_probability",
            ]
        )

    return pd.DataFrame([vars(r) for r in results])
