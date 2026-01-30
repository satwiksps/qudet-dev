import pytest
import numpy as np
from qudet.analytics.timeseries import (
    QuantumTimeSeriesPredictor,
    QuantumOutlierDetection,
    QuantumDimensionalityReduction
)


class TestQuantumTimeSeriesPredictor:
    """Test suite for Quantum Time Series Predictor."""

    def test_initialization(self):
        """Test time series predictor initialization."""
        predictor = QuantumTimeSeriesPredictor(lookback=5, horizon=1)
        assert predictor.lookback == 5
        assert predictor.horizon == 1

    def test_fit(self, time_series_data):
        """Test time series fitting."""
        predictor = QuantumTimeSeriesPredictor(lookback=3, horizon=1)
        predictor.fit(time_series_data[:, 0], epochs=2)
        
        assert predictor.model_params is not None

    def test_predict_single_step(self, time_series_data):
        """Test single-step prediction."""
        predictor = QuantumTimeSeriesPredictor(lookback=3)
        predictor.fit(time_series_data[:, 0], epochs=1)
        
        predictions = predictor.predict(time_series_data[:, 0], n_steps=1)
        
        assert len(predictions) == 1

    def test_predict_multi_step(self, time_series_data):
        """Test multi-step prediction."""
        predictor = QuantumTimeSeriesPredictor(lookback=3)
        predictor.fit(time_series_data[:, 0], epochs=1)
        
        predictions = predictor.predict(time_series_data[:, 0], n_steps=5)
        
        assert len(predictions) == 5

    def test_create_sequences(self, time_series_data):
        """Test sequence creation."""
        predictor = QuantumTimeSeriesPredictor(lookback=5, horizon=2)
        X, y = predictor._create_sequences(time_series_data[:, 0])
        
        assert len(X) == len(y)
        assert X.shape[1] == 5
        assert y.shape[1] == 2


class TestQuantumOutlierDetection:
    """Test suite for Quantum Outlier Detection."""

    def test_initialization(self):
        """Test outlier detection initialization."""
        detector = QuantumOutlierDetection(threshold=2.0)
        assert detector.threshold == 2.0

    def test_fit(self, clean_data_2d):
        """Test fitting outlier detector."""
        detector = QuantumOutlierDetection()
        detector.fit(clean_data_2d)
        
        assert detector.mean is not None
        assert detector.std is not None

    def test_predict(self, clean_data_2d):
        """Test outlier detection prediction."""
        detector = QuantumOutlierDetection(threshold=3.0)
        detector.fit(clean_data_2d)
        
        outliers = detector.predict(clean_data_2d)
        
        assert len(outliers) == len(clean_data_2d)
        assert all(isinstance(o, (bool, np.bool_)) for o in outliers)

    def test_score(self, clean_data_2d):
        """Test anomaly scoring."""
        detector = QuantumOutlierDetection()
        detector.fit(clean_data_2d)
        
        scores = detector.score(clean_data_2d)
        
        assert len(scores) == len(clean_data_2d)
        assert all(s >= 0 for s in scores)

    def test_outlier_detection_with_outliers(self):
        """Test detection with actual outliers."""
        normal_data = np.random.randn(50, 2)
        outlier = np.array([[10.0, 10.0]])
        
        detector = QuantumOutlierDetection(threshold=2.0)
        detector.fit(normal_data)
        
        is_outlier = detector.predict(outlier)
        
        assert is_outlier[0] == True

    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error before fitting."""
        detector = QuantumOutlierDetection()
        
        with pytest.raises(ValueError):
            detector.predict(np.random.rand(5, 2))


class TestQuantumDimensionalityReduction:
    """Test suite for Quantum Dimensionality Reduction."""

    def test_initialization(self):
        """Test dimensionality reduction initialization."""
        reducer = QuantumDimensionalityReduction(n_components=2, iterations=5)
        assert reducer.n_components == 2
        assert reducer.iterations == 5

    def test_fit(self, clean_data_high_dim):
        """Test fitting reduction model."""
        reducer = QuantumDimensionalityReduction(n_components=2, iterations=2)
        reducer.fit(clean_data_high_dim)
        
        assert reducer.projection_matrix is not None

    def test_transform(self, clean_data_high_dim):
        """Test dimensionality reduction."""
        reducer = QuantumDimensionalityReduction(n_components=2, iterations=2)
        reducer.fit(clean_data_high_dim)
        
        reduced = reducer.transform(clean_data_high_dim)
        
        assert reduced.shape[0] == clean_data_high_dim.shape[0]
        assert reduced.shape[1] == 2

    def test_fit_transform(self, clean_data_high_dim):
        """Test fit and transform in one step."""
        reducer = QuantumDimensionalityReduction(n_components=3, iterations=2)
        
        reduced = reducer.fit_transform(clean_data_high_dim)
        
        assert reduced.shape[0] == clean_data_high_dim.shape[0]
        assert reduced.shape[1] == 3

    def test_different_components(self, clean_data_high_dim):
        """Test with different component counts."""
        n_features = clean_data_high_dim.shape[1]
        
        reducer_1 = QuantumDimensionalityReduction(n_components=1, iterations=1)
        reducer_5 = QuantumDimensionalityReduction(n_components=5, iterations=1)
        
        reduced_1 = reducer_1.fit_transform(clean_data_high_dim)
        reduced_5 = reducer_5.fit_transform(clean_data_high_dim)
        
        assert reduced_1.shape[1] == 1
        assert reduced_5.shape[1] == 5

    def test_transform_before_fit_raises_error(self):
        """Test that transform raises error before fitting."""
        reducer = QuantumDimensionalityReduction()
        
        with pytest.raises(ValueError):
            reducer.transform(np.random.rand(10, 5))
