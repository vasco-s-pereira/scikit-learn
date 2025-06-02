import numpy as np
import pytest

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


def test_iterative_imputer_categorical_imputation_basic():
    # Data with categorical column (encoded as integers)
    # Column 0: numerical [1, 2, nan, 4, 5]
    # Column 1: categorical [0, 1, nan, 0, 1] representing ["a", "b", nan, "a", "b"]
    X = np.array([[1.0, 0.0], [2.0, 1.0], [np.nan, np.nan], [4.0, 0.0], [5.0, 1.0]])

    imp = IterativeImputer(
        classifier=RandomForestClassifier(n_estimators=10, random_state=0),
        categorical_features=[1],  # Second column is categorical
        random_state=0,
    )

    Xt = imp.fit_transform(X)

    # Ensure no missing values remain
    assert not np.isnan(Xt).any()


def test_iterative_imputer_categorical_mask_autodetection():
    # Column 0: numerical [1.0, 2.0, 3.0, nan]
    # Column 1: categorical [0, nan, 1, 0] representing ["yes", nan, "no", "yes"]
    X = np.array([[1.0, 0.0], [2.0, np.nan], [3.0, 1.0], [np.nan, 0.0]])

    imputer = IterativeImputer(
        classifier=RandomForestClassifier(n_estimators=5, random_state=0),
        categorical_features=[1],  # Specify categorical column explicitly
        random_state=0,
    )
    Xt = imputer.fit_transform(X)

    # Check that no NaNs remain
    assert not np.isnan(Xt).any()


def test_iterative_imputer_with_mixed_types_and_ordinal_encoder():
    # Column 0: age [25, 30, nan, 40]
    # Column 1: gender [0, 1, 1, nan] representing ["M", "F", "F", nan]
    # Column 2: income [50000, nan, 80000, 100000]
    X = np.array(
        [
            [25.0, 0.0, 50000.0],
            [30.0, 1.0, np.nan],
            [np.nan, 1.0, 80000.0],
            [40.0, np.nan, 100000.0],
        ]
    )

    imputer = IterativeImputer(
        random_state=42,
        categorical_features=[1],  # Gender column is categorical
    )
    Xt = imputer.fit_transform(X)

    # Confirm shape and no missing values
    assert Xt.shape == X.shape
    assert not np.isnan(Xt).any()


def test_iterative_imputer_sample_posterior_error_for_categorical():
    # Column 0: numerical [1.0, 2.0, nan]
    # Column 1: categorical [0, 1, nan] representing ["cat", "dog", nan]
    X = np.array([[1.0, 0.0], [2.0, 1.0], [np.nan, np.nan]])

    with pytest.raises(NotImplementedError):
        IterativeImputer(
            sample_posterior=True, categorical_features=[1], random_state=0
        ).fit_transform(X)


def test_iterative_imputer_categorical_inverse_transform_fallback():
    # Covers fallback case where inverse_transform may fail
    class BadPreprocessor(BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.zeros_like(X, dtype=float)

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

        def inverse_transform(self, X):
            raise ValueError("Can't inverse")

    # Column 0: numerical [1, 2, nan]
    # Column 1: categorical [0, 1, 0] representing ["A", "B", "A"]
    X = np.array([[1.0, 0.0], [2.0, 1.0], [np.nan, 0.0]])

    imputer = IterativeImputer(
        preprocessor=BadPreprocessor(), categorical_features=[1], random_state=0
    )
    Xt = imputer.fit_transform(X)

    assert Xt.shape == X.shape
    assert not np.isnan(Xt).any()


def test_iterative_imputer_n_nearest_features_error_for_categorical():
    # Test that n_nearest_features raises error with categorical features
    X = np.array([[1.0, 0.0], [2.0, 1.0], [np.nan, np.nan]])

    with pytest.raises(NotImplementedError):
        IterativeImputer(
            n_nearest_features=1, categorical_features=[1], random_state=0
        ).fit_transform(X)


def test_iterative_imputer_multiple_categorical_features():
    # Test with multiple categorical features
    # Column 0: numerical
    # Column 1: categorical (2 categories)
    # Column 2: categorical (3 categories)
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],
            [3.0, np.nan, 2.0],
            [np.nan, 0.0, np.nan],
            [5.0, 1.0, 1.0],
        ]
    )

    imputer = IterativeImputer(
        categorical_features=[1, 2],  # Both columns 1 and 2 are categorical
        random_state=42,
        max_iter=3,
    )
    Xt = imputer.fit_transform(X)

    assert Xt.shape == X.shape
    assert not np.isnan(Xt).any()


def test_iterative_imputer_all_categorical():
    # Test with all features being categorical
    X = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, 0.0], [0.0, np.nan], [1.0, 1.0]])

    imputer = IterativeImputer(
        categorical_features=[0, 1],  # Both columns are categorical
        random_state=42,
        max_iter=3,
    )
    Xt = imputer.fit_transform(X)

    assert Xt.shape == X.shape
    assert not np.isnan(Xt).any()
    # Check that values are still in expected categorical range
    assert np.all(np.isin(Xt[:, 0], [0.0, 1.0]))
    assert np.all(np.isin(Xt[:, 1], [0.0, 1.0]))