# Advanced Time Series Forecasting

Industrial-grade time series forecasting with ARIMA, LSTM, and evaluation metrics.

## Features

- **ARIMA Baseline**: Classical statistical forecasting
- **LSTM Deep Learning**: Neural network-based forecasting
- **Model Comparison**: Side-by-side ARIMA vs LSTM with MAE/RMSE
- **Reproducible**: Executed notebooks with saved outputs

## Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| `forecast_model.ipynb` | LSTM baseline forecasting pipeline | ✅ |
| `02_arima_vs_lstm_forecasting.ipynb` | ARIMA vs LSTM comparison | ✅ Executed |

## Usage

```python
from forecasting import forecast_lstm, evaluate_model

# Train LSTM model
model = forecast_lstm(data, epochs=100)

# Evaluate
mae, rmse = evaluate_model(model, test_data)
```

## Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)

## License

MIT
