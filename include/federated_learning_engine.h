#ifndef FEDERATED_LEARNING_ENGINE_H
#define FEDERATED_LEARNING_ENGINE_H

#include <vector>
#include <string>
#include <map>
#include <memory>

namespace Forecast {

struct ClientData {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::string client_id;
};

struct FederatedUpdate {
    std::vector<double> weights;
    double loss;
    int round;
    std::string client_id;
};

class FederatedLearningEngine {
public:
    FederatedLearningEngine();
    ~FederatedLearningEngine();
    
    bool initialize(int num_clients, int rounds);
    
    // Client registration
    void register_client(const std::string& client_id, 
                       const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y);
    
    // Training rounds
    void perform_federated_round(int round_number);
    void aggregate_updates(const std::vector<FederatedUpdate>& updates);
    
    // Model management
    void set_global_model(const std::vector<double>& weights);
    std::vector<double> get_global_model();
    
    // Advanced federated methods
    void enable_differential_privacy(bool enable, double epsilon = 1.0);
    void enable_secure_aggregation(bool enable);
    void enable_compression(bool enable);
    
    // Evaluation
    double evaluate_global_model(const std::vector<std::vector<double>>& X, 
                              const std::vector<double>& y);
    
    void generate_federated_report();
    
private:
    bool initialized_;
    int num_clients_;
    int total_rounds_;
    int current_round_;
    
    std::map<std::string, ClientData> clients_;
    std::vector<double> global_weights_;
    
    bool differential_privacy_enabled_;
    double privacy_epsilon_;
    bool secure_aggregation_enabled_;
    bool compression_enabled_;
    
    std::vector<double> add_noise(const std::vector<double>& weights, double epsilon);
    std::vector<double> compress_weights(const std::vector<double>& weights);
    std::vector<double> decompress_weights(const std::vector<double>& compressed);
};

} // namespace Forecast

#endif // FEDERATED_LEARNING_ENGINE_H
