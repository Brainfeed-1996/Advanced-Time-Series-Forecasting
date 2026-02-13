"""
Tests for Time Series Forecasting.
"""

import pytest
import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def simple_moving_average(data, window=3):
    """Simple moving average forecast."""
    return np.convolve(data, np.ones(window)/window, mode='valid')


class TestMetrics:
    """Test cases for forecasting metrics."""
    
    def test_mae_identical(self):
        """Test MAE with identical arrays."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert mae(y_true, y_pred) == 0.0
    
    def test_mae_simple(self):
        """Test MAE calculation."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        assert mae(y_true, y_pred) == pytest.approx(1/3)
    
    def test_rmse_identical(self):
        """Test RMSE with identical arrays."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        assert rmse(y_true, y_pred) == 0.0
    
    def test_rmse_simple(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(1/3))
    
    def test_mape_identical(self):
        """Test MAPE with identical arrays."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])
        assert mape(y_true, y_pred) == 0.0
    
    def test_mape_simple(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 300])
        expected = np.mean([10, 10, 0])
        assert mape(y_true, y_pred) == pytest.approx(expected)


class TestMovingAverage:
    """Test cases for moving average forecast."""
    
    def test_sma_length(self):
        """Test SMA output length."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = simple_moving_average(data, window=3)
        assert len(result) == len(data) - 3 + 1
    
    def test_sma_values(self):
        """Test SMA calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = simple_moving_average(data, window=2)
        expected = np.array([1.5, 2.5, 3.5, 4.5])
        assert np.allclose(result, expected)


class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_train_test_split(self):
        """Test simple train-test split."""
        data = np.arange(100)
        train_size = int(len(data) * 0.8)
        train = data[:train_size]
        test = data[train_size:]
        
        assert len(train) == 80
        assert len(test) == 20
        assert train[-1] == 79
        assert test[0] == 80


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
