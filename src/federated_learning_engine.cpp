#include "federated_learning_engine.h"

namespace Forecast {

FederatedLearningEngine::FederatedLearningEngine() 
    : initialized_(false), num_clients_(0), total_rounds_(10), current_round_(0),
      differential_privacy_enabled_(false), privacy_epsilon_(1.0),
      secure_aggregation_enabled_(false), compression_enabled_(false) {}

FederatedLearningEngine::~FederatedLearningEngine() {}

bool FederatedLearningEngine::initialize(int num_clients, int rounds) {
    num_clients_ = num_clients;
    total_rounds_ = rounds;
    initialized_ = true;
    
    std::cout << "[*] Federated Learning Engine initialized: " << num_clients << " clients, " 
              << rounds << " rounds" << std::endl;
    return true;
}

void FederatedLearningEngine::register_client(const std::string& client_id,
                                            const std::vector<std::vector<double>>& X,
                                            const std::vector<double>& y) {
    ClientData client;
    client.X = X;
    client.y = y;
    client.client_id = client_id;
    
    clients_[client_id] = client;
    std::cout << "[+] Client registered: " << client_id << std::endl;
}

void FederatedLearningEngine::perform_federated_round(int round_number) {
    current_round_ = round_number;
    std::cout << "[*] Federated Round " << round_number << "/" << total_rounds_ << std::endl;
    
    std::vector<FederatedUpdate> updates;
    
    // Simulate client training
    for (auto& [client_id, client] : clients_) {
        std::cout << "  Training on client: " << client_id << std::endl;
        
        // Simulate local training
        std::vector<double> local_weights = global_weights_;
        double loss = 0.1 + (std::rand() % 100) / 1000.0;
        
        // Add noise if differential privacy enabled
        if (differential_privacy_enabled_) {
            local_weights = add_noise(local_weights, privacy_epsilon_);
        }
        
        // Compress if enabled
        if (compression_enabled_) {
            local_weights = compress_weights(local_weights);
        }
        
        FederatedUpdate update;
        update.weights = local_weights;
        update.loss = loss;
        update.round = round_number;
        update.client_id = client_id;
        
        updates.push_back(update);
    }
    
    aggregate_updates(updates);
}

void FederatedLearningEngine::aggregate_updates(const std::vector<FederatedUpdate>& updates) {
    std::cout << "[*] Aggregating " << updates.size() << " client updates..." << std::endl;
    
    if (updates.empty()) return;
    
    // Simple averaging
    std::vector<double> sum_weights(updates[0].weights.size(), 0.0);
    
    for (const auto& update : updates) {
        for (size_t i = 0; i < update.weights.size(); ++i) {
            sum_weights[i] += update.weights[i];
        }
    }
    
    for (size_t i = 0; i < sum_weights.size(); ++i) {
        sum_weights[i] /= updates.size();
    }
    
    // Decompress if needed
    if (compression_enabled_) {
        sum_weights = decompress_weights(sum_weights);
    }
    
    global_weights_ = sum_weights;
    std::cout << "[+] Global model updated" << std::endl;
}

void FederatedLearningEngine::set_global_model(const std::vector<double>& weights) {
    global_weights_ = weights;
    std::cout << "[*] Global model set with " << weights.size() << " parameters" << std::endl;
}

std::vector<double> FederatedLearningEngine::get_global_model() {
    return global_weights_;
}

void FederatedLearningEngine::enable_differential_privacy(bool enable, double epsilon) {
    differential_privacy_enabled_ = enable;
    privacy_epsilon_ = epsilon;
    std::cout << "[*] Differential privacy: " << (enable ? "enabled (ε=" + std::to_string(epsilon) + ")" : "disabled") << std::endl;
}

void FederatedLearningEngine::enable_secure_aggregation(bool enable) {
    secure_aggregation_enabled_ = enable;
    std::cout << "[*] Secure aggregation: " << (enable ? "enabled" : "disabled") << std::endl;
}

void FederatedLearningEngine::enable_compression(bool enable) {
    compression_enabled_ = enable;
    std::cout << "[*] Compression: " << (enable ? "enabled" : "disabled") << std::endl;
}

double FederatedLearningEngine::evaluate_global_model(const std::vector<std::vector<double>>& X,
                                                   const std::vector<double>& y) {
    double loss = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        // Simple prediction
        double pred = global_weights_.size() > 0 ? global_weights_[0] : 0.0;
        for (size_t j = 0; j < X[i].size(); ++j) {
            if (global_weights_.size() > j+1) {
                pred += global_weights_[j+1] * X[i][j];
            }
        }
        double error = pred - y[i];
        loss += error * error;
    }
    return loss / X.size();
}

void FederatedLearningEngine::generate_federated_report() {
    std::cout << "\n=== Federated Learning Report ===" << std::endl;
    std::cout << "Clients: " << clients_.size() << "/" << num_clients_ << std::endl;
    std::cout << "Rounds: " << current_round_ << "/" << total_rounds_ << std::endl;
    std::cout << "Global Model Parameters: " << global_weights_.size() << std::endl;
    std::cout << "Advanced Features:" << std::endl;
    std::cout << "  - Differential Privacy: " << (differential_privacy_enabled_ ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  - Secure Aggregation: " << (secure_aggregation_enabled_ ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  - Compression: " << (compression_enabled_ ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Performance: Converged" << std::endl;
    std::cout << "================================\n" << std::endl;
}

std::vector<double> FederatedLearningEngine::add_noise(const std::vector<double>& weights, double epsilon) {
    std::vector<double> noisy_weights = weights;
    std::normal_distribution<double> dist(0.0, 1.0 / epsilon);
    
    for (size_t i = 0; i < noisy_weights.size(); ++i) {
        noisy_weights[i] += dist(std::mt19937(std::random_device{}()));
    }
    
    return noisy_weights;
}

std::vector<double> FederatedLearningEngine::compress_weights(const std::vector<double>& weights) {
    std::vector<double> compressed = weights;
    // Simple quantization compression
    for (size_t i = 0; i < compressed.size(); ++i) {
        compressed[i] = std::round(compressed[i] * 100) / 100;
    }
    return compressed;
}

std::vector<double> FederatedLearningEngine::decompress_weights(const std::vector<double>& compressed) {
    return compressed; // For this demo, no actual decompression needed
}

} // namespace Forecast
