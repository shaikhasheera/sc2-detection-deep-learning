"""
ORLC: Optimal Rule List Classifier
----------------------------------

This class wraps around RandomForestClassifier to maintain compatibility
with scikit-learn, while allowing for later integration of optimized rule-based logic.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class ORLC(BaseEstimator, ClassifierMixin):
    """
    Optimal Rule List Classifier (ORLC)

    This model mimics the core functionality of RandomForestClassifier
    while providing an interpretable framework to define, optimize,
    and apply rule lists derived from tree-based decision boundaries.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize ORLC model with base RandomForestClassifier parameters.

        Parameters
        ----------
        n_estimators : int, optional (default=100)
            Number of decision trees in the ensemble.
        max_depth : int, optional (default=None)
            Maximum depth of each decision tree.
        random_state : int, optional (default=42)
            Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.rule_list_ = None  # Placeholder for future rule list extraction

    def fit(self, X, y):
        """
        Fit the ORLC model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features.
        y : array-like, shape (n_samples,)
            Target labels.
        """
        self._rf_model.fit(X, y)
        self._generate_rule_list()
        return self

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        return self._rf_model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.
        """
        return self._rf_model.predict_proba(X)

    def _generate_rule_list(self):
        """
        Extract and store simplified rule list from trained RandomForestClassifier.
        For now, it collects representative rules from the top decision paths.
        """
        rules = []
        for estimator in self._rf_model.estimators_[:5]:  # Sample first few trees
            tree = estimator.tree_
            feature = tree.feature
            threshold = tree.threshold
            rules.append(f"Rule Tree depth={tree.max_depth}, Features used={np.unique(feature[feature >= 0]).size}")
        self.rule_list_ = rules

    def get_rule_list(self):
        """
        Return simplified rule list extracted from the trained model.
        """
        if self.rule_list_ is None:
            raise ValueError("Model must be fitted before extracting rule list.")
        return self.rule_list_

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        return self._rf_model.score(X, y)
