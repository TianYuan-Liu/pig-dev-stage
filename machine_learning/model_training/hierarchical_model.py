"""
Hierarchical classification model for tissue-aware developmental stage prediction.
Respects biological differences between tissues while identifying conserved patterns.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class TissueAwareHierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """
    Two-stage hierarchical classifier:
    1. First identifies tissue type
    2. Then applies tissue-specific developmental stage model

    This respects the biological reality that different tissues have
    different developmental programs.
    """

    def __init__(
        self,
        tissue_classifier_params: Optional[Dict] = None,
        stage_classifier_params: Optional[Dict] = None,
        use_conserved_features: bool = True,
        n_conserved_features: int = 100,
        random_state: int = 42
    ):
        """
        Initialize hierarchical classifier.

        Args:
            tissue_classifier_params: Parameters for tissue type classifier
            stage_classifier_params: Parameters for stage classifiers
            use_conserved_features: Whether to identify conserved features
            n_conserved_features: Number of conserved features to select
            random_state: Random seed for reproducibility
        """
        self.tissue_classifier_params = tissue_classifier_params or {
            'max_iter': 1000,
            'random_state': random_state
        }
        self.stage_classifier_params = stage_classifier_params or {
            'max_iter': 500,
            'random_state': random_state
        }
        self.use_conserved_features = use_conserved_features
        self.n_conserved_features = n_conserved_features
        self.random_state = random_state

        # Will be populated during fit
        self.tissue_classifier = None
        self.stage_classifiers = {}
        self.tissue_encoder = LabelEncoder()
        self.stage_encoders = {}
        self.conserved_features = None
        self.tissue_specific_features = {}
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, tissues: np.ndarray,
            feature_names: Optional[np.ndarray] = None) -> 'TissueAwareHierarchicalClassifier':
        """
        Fit the hierarchical classifier.

        Args:
            X: Feature matrix (samples x features)
            y: Developmental stage labels
            tissues: Tissue type for each sample
            feature_names: Optional gene names for features

        Returns:
            Self for chaining
        """
        logger.info("Training hierarchical tissue-aware classifier")

        self.feature_names = feature_names

        # Encode tissues
        tissues_encoded = self.tissue_encoder.fit_transform(tissues)
        unique_tissues = self.tissue_encoder.classes_

        logger.info(f"Found {len(unique_tissues)} unique tissues: {unique_tissues}")

        # Step 1: Train tissue classifier
        logger.info("Training tissue type classifier...")
        self.tissue_classifier = LogisticRegression(**self.tissue_classifier_params)
        self.tissue_classifier.fit(X, tissues_encoded)
        tissue_acc = accuracy_score(tissues_encoded, self.tissue_classifier.predict(X))
        logger.info(f"Tissue classifier accuracy: {tissue_acc:.3f}")

        # Step 2: Identify conserved features if requested
        if self.use_conserved_features:
            self._identify_conserved_features(X, y, tissues, unique_tissues)

        # Step 3: Train tissue-specific stage classifiers
        logger.info("Training tissue-specific stage classifiers...")
        for tissue in unique_tissues:
            tissue_mask = tissues == tissue
            X_tissue = X[tissue_mask]
            y_tissue = y[tissue_mask]

            if len(np.unique(y_tissue)) < 2:
                logger.warning(f"Skipping {tissue}: insufficient stage diversity")
                continue

            # Encode stages for this tissue
            self.stage_encoders[tissue] = LabelEncoder()
            y_tissue_encoded = self.stage_encoders[tissue].fit_transform(y_tissue)

            # Select features for this tissue
            if self.use_conserved_features and self.conserved_features is not None:
                # Combine conserved and tissue-specific features
                tissue_features = self._select_tissue_specific_features(
                    X_tissue, y_tissue_encoded, tissue
                )
                features_to_use = np.union1d(self.conserved_features, tissue_features)
                X_tissue_selected = X_tissue[:, features_to_use]
            else:
                X_tissue_selected = X_tissue
                features_to_use = np.arange(X.shape[1])

            self.tissue_specific_features[tissue] = features_to_use

            # Train classifier for this tissue
            self.stage_classifiers[tissue] = LogisticRegression(**self.stage_classifier_params)
            self.stage_classifiers[tissue].fit(X_tissue_selected, y_tissue_encoded)

            # Evaluate within-tissue performance
            stage_acc = accuracy_score(
                y_tissue_encoded,
                self.stage_classifiers[tissue].predict(X_tissue_selected)
            )
            logger.info(f"  {tissue} stage classifier accuracy: {stage_acc:.3f}")

        return self

    def predict(self, X: np.ndarray, known_tissues: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict developmental stages using hierarchical approach.

        Args:
            X: Feature matrix (samples x features)
            known_tissues: Optional known tissue types (bypasses tissue prediction)

        Returns:
            Predicted developmental stages
        """
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=object)

        # Step 1: Predict or use known tissue types
        if known_tissues is not None:
            tissue_predictions = self.tissue_encoder.transform(known_tissues)
            tissue_names = known_tissues
        else:
            tissue_predictions = self.tissue_classifier.predict(X)
            tissue_names = self.tissue_encoder.inverse_transform(tissue_predictions)

        # Step 2: Apply tissue-specific stage classifiers
        for tissue in np.unique(tissue_names):
            if tissue not in self.stage_classifiers:
                logger.warning(f"No stage classifier for tissue: {tissue}")
                continue

            tissue_mask = tissue_names == tissue
            if not np.any(tissue_mask):
                continue

            X_tissue = X[tissue_mask]

            # Use tissue-specific features
            features = self.tissue_specific_features.get(tissue, np.arange(X.shape[1]))
            X_tissue_selected = X_tissue[:, features]

            # Predict stages
            stage_predictions = self.stage_classifiers[tissue].predict(X_tissue_selected)
            stage_names = self.stage_encoders[tissue].inverse_transform(stage_predictions)

            predictions[tissue_mask] = stage_names

        return predictions

    def predict_proba(self, X: np.ndarray, known_tissues: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Predict probabilities for developmental stages.

        Returns dict with tissue names as keys and probability arrays as values.
        """
        probabilities = {}

        # Get tissue predictions
        if known_tissues is not None:
            tissue_names = known_tissues
        else:
            tissue_predictions = self.tissue_classifier.predict(X)
            tissue_names = self.tissue_encoder.inverse_transform(tissue_predictions)

        for tissue in np.unique(tissue_names):
            if tissue not in self.stage_classifiers:
                continue

            tissue_mask = tissue_names == tissue
            if not np.any(tissue_mask):
                continue

            X_tissue = X[tissue_mask]
            features = self.tissue_specific_features.get(tissue, np.arange(X.shape[1]))
            X_tissue_selected = X_tissue[:, features]

            probabilities[tissue] = self.stage_classifiers[tissue].predict_proba(X_tissue_selected)

        return probabilities

    def _identify_conserved_features(self, X: np.ndarray, y: np.ndarray,
                                    tissues: np.ndarray, unique_tissues: np.ndarray):
        """
        Identify features with consistent developmental patterns across tissues.
        """
        logger.info(f"Identifying {self.n_conserved_features} conserved features...")

        from sklearn.feature_selection import f_classif

        # Calculate feature importance for each tissue
        feature_scores = []

        for tissue in unique_tissues:
            tissue_mask = tissues == tissue
            if np.sum(tissue_mask) < 20:  # Skip small tissues
                continue

            X_tissue = X[tissue_mask]
            y_tissue = y[tissue_mask]

            # Skip if insufficient diversity
            if len(np.unique(y_tissue)) < 2:
                continue

            # Calculate F-scores
            scores, _ = f_classif(X_tissue, y_tissue)
            feature_scores.append(scores)

        if len(feature_scores) == 0:
            logger.warning("Could not identify conserved features")
            self.conserved_features = None
            return

        # Find features with consistent high scores across tissues
        feature_scores = np.array(feature_scores)

        # Use mean score weighted by consistency (low std)
        mean_scores = np.mean(feature_scores, axis=0)
        std_scores = np.std(feature_scores, axis=0)

        # Combine mean and consistency (lower std is better)
        consistency_weight = 1 / (1 + std_scores)
        combined_scores = mean_scores * consistency_weight

        # Select top features
        n_features = min(self.n_conserved_features, len(combined_scores))
        self.conserved_features = np.argsort(combined_scores)[-n_features:]

        logger.info(f"Selected {len(self.conserved_features)} conserved features")

        if self.feature_names is not None:
            conserved_names = self.feature_names[self.conserved_features][:10]
            logger.info(f"Top conserved genes: {conserved_names}")

    def _select_tissue_specific_features(self, X: np.ndarray, y: np.ndarray,
                                        tissue: str, n_features: int = 100) -> np.ndarray:
        """
        Select tissue-specific features that complement conserved features.
        """
        from sklearn.feature_selection import SelectKBest, f_classif

        # Select features specific to this tissue
        selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)

        # Get indices of selected features
        tissue_features = np.where(selector.get_support())[0]

        # Remove overlap with conserved features if they exist
        if self.conserved_features is not None:
            tissue_features = np.setdiff1d(tissue_features, self.conserved_features)

        return tissue_features[:n_features]

    def get_feature_importance(self, tissue: str) -> pd.DataFrame:
        """
        Get feature importance for a specific tissue.

        Args:
            tissue: Tissue name

        Returns:
            DataFrame with feature names and importance scores
        """
        if tissue not in self.stage_classifiers:
            raise ValueError(f"No model for tissue: {tissue}")

        classifier = self.stage_classifiers[tissue]
        features = self.tissue_specific_features[tissue]

        # Get coefficients (for logistic regression)
        if hasattr(classifier, 'coef_'):
            # Average absolute coefficients across classes
            importance = np.mean(np.abs(classifier.coef_), axis=0)
        else:
            # For other classifiers, use feature_importances_
            importance = classifier.feature_importances_

        # Create DataFrame
        if self.feature_names is not None:
            feature_names = self.feature_names[features]
        else:
            feature_names = [f"Feature_{i}" for i in features]

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'is_conserved': [i in self.conserved_features for i in features]
        })

        return df.sort_values('importance', ascending=False)

    def save_model(self, path: Union[str, Path]):
        """Save the trained model and metadata."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'tissue_classifier': self.tissue_classifier,
            'stage_classifiers': self.stage_classifiers,
            'tissue_encoder': self.tissue_encoder,
            'stage_encoders': self.stage_encoders,
            'conserved_features': self.conserved_features,
            'tissue_specific_features': self.tissue_specific_features,
            'feature_names': self.feature_names,
            'params': {
                'tissue_classifier_params': self.tissue_classifier_params,
                'stage_classifier_params': self.stage_classifier_params,
                'use_conserved_features': self.use_conserved_features,
                'n_conserved_features': self.n_conserved_features,
                'random_state': self.random_state
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'TissueAwareHierarchicalClassifier':
        """Load a saved model."""
        import pickle

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance with saved parameters
        model = cls(**model_data['params'])

        # Restore fitted attributes
        model.tissue_classifier = model_data['tissue_classifier']
        model.stage_classifiers = model_data['stage_classifiers']
        model.tissue_encoder = model_data['tissue_encoder']
        model.stage_encoders = model_data['stage_encoders']
        model.conserved_features = model_data['conserved_features']
        model.tissue_specific_features = model_data['tissue_specific_features']
        model.feature_names = model_data['feature_names']

        return model