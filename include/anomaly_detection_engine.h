#ifndef ANOMALY_DETECTION_ENGINE_H
#define ANOMALY_DETECTION_ENGINE_H

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <random>

namespace Forecast {

struct AnomalyResult {
    std::vector<int> anomaly_indices;
    std::vector<double> anomaly_scores;
    std::vector<double> thresholds;
    double global_score;
    std::string detection_method;
    std::vector<double> raw_scores;
    std::vector<bool> is_anomaly;
};

struct DetectionConfig {
    double sensitivity;
    double threshold_percentile;
    int window_size;
    bool use_ensemble;
    std::vector<std::string> methods;
};

class AnomalyDetectionEngine {
public:
    AnomalyDetectionEngine();
    ~AnomalyDetectionEngine();
    
    bool initialize(const DetectionConfig& config);
    
    // Statistical methods
    AnomalyResult detect_zscore(const std::vector<double>& data, double threshold = 3.0);
    AnomalyResult detect_iqr(const std::vector<double>& data, double multiplier = 1.5);
    AnomalyResult detect_modified_zscore(const std::vector<double>& data, double threshold = 3.5);
    
    // Machine learning methods
    AnomalyResult detect_isolation_forest(const std::vector<double>& data, int n_trees = 100);
    AnomalyResult detect_local_outlier_factor(const std::vector<double>& data, int n_neighbors = 20);
    AnomalyResult detect_one_class_svm(const std::vector<double>& data, double nu = 0.1);
    
    // Deep learning methods
    AnomalyResult detect_autoencoder(const std::vector<std::vector<double>>& data, double threshold = 0.1);
    AnomalyResult detect_lstm_autoencoder(const std::vector<std::vector<double>>& sequences, double threshold = 0.1);
    
    // Ensemble methods
    AnomalyResult detect_ensemble(const std::vector<double>& data);
    AnomalyResult detect_ensemble_vote(const std::vector<double>& data);
    
    // Time series specific
    AnomalyResult detect_seasonal_decomposition(const std::vector<double>& data, int period = 7);
    AnomalyResult detect_change_point(const std::vector<double>& data, double change_threshold = 0.5);
    AnomalyResult detect_gradual_change(const std::vector<double>& data, int window = 10);
    
    // Streaming methods
    AnomalyResult detect_streaming(const std::vector<double>& data, double sensitivity = 2.0);
    void update_streaming_model(double new_value);
    
    // Evaluation
    double calculate_precision(const std::vector<bool>& predicted, const std::vector<bool>& actual);
    double calculate_recall(const std::vector<bool>& predicted, const std::vector<bool>& actual);
    double calculate_f1_score(const std::vector<bool>& predicted, const std::vector<bool>& actual);
    
    // Analysis
    void analyze_anomaly_patterns(const AnomalyResult& result);
    void visualize_anomalies(const std::vector<double>& data, const AnomalyResult& result);
    
    void generate_anomaly_report();
    
private:
    bool initialized_;
    DetectionConfig config_;
    
    std::vector<double> streaming_buffer_;
    std::map<std::string, double> stats_;
    
    std::mt19937 rng_;
    
    // Internal methods
    std::vector<double> calculate_rolling_stats(const std::vector<double>& data, int window);
    std::vector<double> calculate_median_absolute_deviation(const std::vector<double>& data);
    double calculate_moving_average(const std::vector<double>& data, int window, int position);
};

} // namespace Forecast

#endif // ANOMALY_DETECTION_ENGINE_H
