import pytest
import numpy as np
from qudet.analytics.ensemble import QuantumEnsemble, QuantumDataAugmentation, QuantumMetaLearner


class TestQuantumEnsemble:
    """Test suite for Quantum Ensemble."""

    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = QuantumEnsemble(n_models=3, voting="majority")
        assert ensemble.n_models == 3
        assert ensemble.voting == "majority"

    def test_fit(self, binary_data):
        """Test ensemble training."""
        X, y = binary_data
        ensemble = QuantumEnsemble(n_models=2)
        
        ensemble.fit(X[:20], y[:20])
        
        assert len(ensemble.models) == 2
        assert all(m["score"] >= 0 for m in ensemble.models)

    def test_predict_majority_voting(self, binary_data):
        """Test prediction with majority voting."""
        X, y = binary_data
        ensemble = QuantumEnsemble(n_models=2, voting="majority")
        ensemble.fit(X[:20], y[:20])
        
        predictions = ensemble.predict(X[20:25])
        
        assert len(predictions) == 5

    def test_predict_weighted_voting(self, binary_data):
        """Test prediction with weighted voting."""
        X, y = binary_data
        ensemble = QuantumEnsemble(n_models=2, voting="weighted")
        ensemble.fit(X[:20], y[:20])
        
        predictions = ensemble.predict(X[20:25])
        
        assert len(predictions) == 5

    def test_weights_computation(self, binary_data):
        """Test weight computation from model scores."""
        X, y = binary_data
        ensemble = QuantumEnsemble(n_models=2)
        ensemble.fit(X[:20], y[:20])
        
        assert np.allclose(np.sum(ensemble.weights), 1.0)
        assert all(w >= 0 for w in ensemble.weights)


class TestQuantumDataAugmentation:
    """Test suite for Quantum Data Augmentation."""

    def test_initialization(self):
        """Test data augmentation initialization."""
        aug = QuantumDataAugmentation(n_qubits=4, augmentation_factor=2)
        assert aug.n_qubits == 4
        assert aug.augmentation_factor == 2

    def test_fit(self, clean_data_2d):
        """Test learning data distribution."""
        aug = QuantumDataAugmentation(augmentation_factor=2)
        aug.fit(clean_data_2d[:50])
        
        assert aug.generative_params is not None
        assert "mean" in aug.generative_params
        assert "std" in aug.generative_params

    def test_transform(self, clean_data_2d):
        """Test data augmentation."""
        aug = QuantumDataAugmentation(augmentation_factor=2)
        aug.fit(clean_data_2d[:50])
        
        augmented = aug.transform(clean_data_2d[50:55])
        
        assert len(augmented) == 15  # 5 original + 5*augmentation_factor
        assert augmented.shape[1] == clean_data_2d.shape[1]

    def test_augmentation_factor(self, clean_data_2d):
        """Test different augmentation factors."""
        original_size = len(clean_data_2d[:10])
        
        aug_2x = QuantumDataAugmentation(augmentation_factor=2)
        aug_2x.fit(clean_data_2d[:20])
        augmented_2x = aug_2x.transform(clean_data_2d[:10])
        
        aug_3x = QuantumDataAugmentation(augmentation_factor=3)
        aug_3x.fit(clean_data_2d[:20])
        augmented_3x = aug_3x.transform(clean_data_2d[:10])
        
        assert len(augmented_2x) == original_size + original_size * 2
        assert len(augmented_3x) == original_size + original_size * 3


class TestQuantumMetaLearner:
    """Test suite for Quantum Meta-Learner."""

    def test_initialization(self):
        """Test meta-learner initialization."""
        ml = QuantumMetaLearner(inner_lr=0.01, outer_lr=0.001)
        assert ml.inner_lr == 0.01
        assert ml.outer_lr == 0.001

    def test_fit(self, binary_data):
        """Test meta-training."""
        X, y = binary_data
        
        X_support = X[:10]
        y_support = y[:10]
        X_query = X[10:20]
        y_query = y[10:20]
        
        ml = QuantumMetaLearner()
        ml.fit(X_support, y_support, X_query, y_query, episodes=2)
        
        assert ml.meta_params is not None

    def test_predict_after_fit(self, binary_data):
        """Test prediction after meta-training."""
        X, y = binary_data
        
        X_support = X[:10]
        y_support = y[:10]
        X_query = X[10:20]
        y_query = y[10:20]
        
        ml = QuantumMetaLearner()
        ml.fit(X_support, y_support, X_query, y_query, episodes=2)
        
        predictions = ml.predict(X[20:25])
        
        assert len(predictions) == 5

    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error before fitting."""
        ml = QuantumMetaLearner()
        
        with pytest.raises(ValueError):
            ml.predict(np.random.rand(5, 4))
