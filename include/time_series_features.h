#ifndef TIME_SERIES_FEATURES_H
#define TIME_SERIES_FEATURES_H

#include <vector>
#include <string>
#include <map>
#include <complex>

namespace Forecast {

struct FeatureSet {
    std::vector<double> features;
    std::vector<std::string> feature_names;
    std::map<std::string, double> statistics;
};

struct FeatureConfig {
    bool enable_statistical_features;
    bool enable_temporal_features;
    bool enable_spectral_features;
    bool enable_entropy_features;
    bool enable_trend_features;
    bool enable_seasonality_features;
    bool enable_volatility_features;
    bool enable_crossing_features;
    int sliding_window_size;
    double sampling_rate;
};

class TimeSeriesFeatures {
public:
    TimeSeriesFeatures();
    ~TimeSeriesFeatures();
    
    bool initialize(const FeatureConfig& config);
    
    // Statistical features
    FeatureSet extract_statistical_features(const std::vector<double>& data);
    FeatureSet extract_moment_features(const std::vector<double>& data);
    FeatureSet extract_distribution_features(const std::vector<double>& data);
    
    // Temporal features
    FeatureSet extract_temporal_features(const std::vector<double>& data);
    FeatureSet extract_autocorrelation_features(const std::vector<double>& data);
    FeatureSet extract_partial_autocorrelation(const std::vector<double>& data);
    
    // Spectral features
    FeatureSet extract_spectral_features(const std::vector<double>& data);
    FeatureSet extract_frequency_domain_features(const std::vector<double>& data);
    std::vector<std::complex<double>> compute_fft(const std::vector<double>& data);
    
    // Entropy features
    FeatureSet extract_entropy_features(const std::vector<double>& data);
    double calculate_sample_entropy(const std::vector<double>& data, int m, double r);
    double calculate_permutation_entropy(const std::vector<double>& data, int order);
    double calculate_approximate_entropy(const std::vector<double>& data, int m, double r);
    
    // Trend features
    FeatureSet extract_trend_features(const std::vector<double>& data);
    FeatureSet extract_change_point_features(const std::vector<double>& data);
    FeatureSet extract_segment_features(const std::vector<double>& data);
    
    // Seasonality features
    FeatureSet extract_seasonality_features(const std::vector<double>& data, int period);
    FeatureSet extract_fourier_features(const std::vector<double>& data, int n_harmonics);
    FeatureSet extract_wavelet_features(const std::vector<double>& data);
    
    // Volatility features
    FeatureSet extract_volatility_features(const std::vector<double>& data);
    FeatureSet extract_range_features(const std::vector<double>& data);
    FeatureSet extract_jump_features(const std::vector<double>& data);
    
    // Crossing features
    FeatureSet extract_crossing_features(const std::vector<double>& data);
    FeatureSet extract_level_crossing_features(const std::vector<double>& data);
    
    // Combined extraction
    FeatureSet extract_all_features(const std::vector<double>& data);
    
    // Feature selection
    std::vector<int> select_features_by_variance(const FeatureSet& features, double threshold);
    std::vector<int> select_features_by_correlation(const FeatureSet& features, double threshold);
    std::vector<int> select_features_by_mutual_information(const FeatureSet& features);
    
    // Normalization
    FeatureSet normalize_features(const FeatureSet& features, const std::string& method = "zscore");
    
    void generate_feature_report();
    
private:
    bool initialized_;
    FeatureConfig config_;
    
    std::vector<double> calculate_fft_magnitude(const std::vector<double>& data);
    std::vector<double> calculate_fft_phase(const std::vector<double>& data);
    std::vector<double> calculate_spectral_density(const std::vector<double>& data);
    int find_optimal_period(const std::vector<double>& data);
};

} // namespace Forecast

#endif // TIME_SERIES_FEATURES_H
