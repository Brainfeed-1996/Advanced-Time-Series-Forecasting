#ifndef TIME_SERIES_FORECASTER_H
#define TIME_SERIES_FORECASTER_H

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <cmath>
#include <random>

namespace Forecast {

struct TimeSeriesData {
    std::vector<double> values;
    std::vector<std::string> timestamps;
    std::string name;
    double frequency; // e.g., 1.0 for daily, 7.0 for weekly
};

struct ForecastResult {
    std::vector<double> predictions;
    std::vector<double> confidence_intervals;
    std::vector<double> residuals;
    double rmse;
    double mae;
    double r_squared;
    std::string model_used;
};

struct ModelConfig {
    int seq_length;
    int forecast_horizon;
    int lstm_units_1;
    int lstm_units_2;
    double dropout_rate;
    double learning_rate;
    int epochs;
    std::string optimizer;
    bool use_attention;
    bool use_bidirectional;
    bool use_residual_connections;
};

class TimeSeriesForecaster {
public:
    TimeSeriesForecaster();
    ~TimeSeriesForecaster();
    
    bool initialize(const ModelConfig& config);
    
    // Data preprocessing
    TimeSeriesData preprocess_data(const std::vector<double>& raw_data, 
                                 const std::vector<std::string>& timestamps = {});
    std::vector<std::vector<double>> create_sequences(const std::vector<double>& data, 
                                                    int seq_length, int forecast_horizon);
    
    // Feature engineering
    std::vector<std::vector<double>> engineer_features(const std::vector<std::vector<double>>& sequences);
    std::vector<double> add_lag_features(const std::vector<double>& series, int lag);
    std::vector<double> calculate_rolling_mean(const std::vector<double>& series, int window);
    std::vector<double> calculate_rolling_std(const std::vector<double>& series, int window);
    
    // Model building
    void build_model();
    void compile_model();
    
    // Training
    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    void train_with_validation(const std::vector<std::vector<double>>& X, 
                             const std::vector<double>& y,
                             double validation_split = 0.2);
    
    // Prediction
    ForecastResult predict(const std::vector<std::vector<double>>& X);
    ForecastResult predict_single(const std::vector<double>& sequence);
    
    // Evaluation
    ForecastResult evaluate(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    void analyze_residuals(const std::vector<double>& residuals);
    
    // Advanced features
    void enable_quantum_inspired_optimization(bool enable);
    void enable_federated_learning(bool enable);
    void enable_online_learning(bool enable);
    void enable_multivariate_forecasting(bool enable);
    
    // Model management
    void save_model(const std::string& filename);
    void load_model(const std::string& filename);
    
    // Statistical analysis
    void perform_statistical_tests(const std::vector<double>& residuals);
    
    void generate_report();
    
private:
    bool initialized_;
    ModelConfig config_;
    
    // Neural network parameters
    std::vector<std::vector<double>> weights_;
    std::vector<std::vector<double>> biases_;
    
    // Data scaling
    double scaler_mean_;
    double scaler_std_;
    
    // Training history
    std::vector<double> training_loss_;
    std::vector<double> validation_loss_;
    
    // Random number generator
    std::mt19937 rng_;
    
    // Internal methods
    std::vector<double> apply_scaling(const std::vector<double>& data);
    std::vector<double> inverse_scaling(const std::vector<double>& scaled_data);
    
    // LSTM operations
    std::vector<double> lstm_cell_forward(const std::vector<double>& input, 
                                        const std::vector<double>& hidden_state,
                                        const std::vector<double>& cell_state);
    
    // Attention mechanism
    std::vector<double> attention_mechanism(const std::vector<std::vector<double>>& lstm_outputs);
    
    // Optimization
    void update_weights(const std::vector<std::vector<double>>& gradients, double learning_rate);
};

} // namespace Forecast

#endif // TIME_SERIES_FORECASTER_H
