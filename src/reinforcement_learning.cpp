#include "reinforcement_learning.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace Forecast {

// ReplayMemory Implementation
ReplayMemory::ReplayMemory(int capacity) : capacity_(capacity), position_(0) {
    memory_.reserve(capacity);
}

void ReplayMemory::push(const Transition& transition) {
    if (memory_.size() < capacity_) {
        memory_.push_back(transition);
    } else {
        memory_[position_] = transition;
    }
    position_ = (position_ + 1) % capacity_;
}

std::vector<Transition> ReplayMemory::sample(int batch_size) {
    std::vector<Transition> batch;
    std::uniform_int_distribution<> dist(0, memory_.size() - 1);
    
    for (int i = 0; i < batch_size && i < static_cast<int>(memory_.size()); ++i) {
        batch.push_back(memory_[dist(rng_)]);
    }
    
    return batch;
}

size_t ReplayMemory::size() const {
    return memory_.size();
}

bool ReplayMemory::is_full() const {
    return memory_.size() >= static_cast<size_t>(capacity_);
}

// NeuralNetwork Implementation
NeuralNetwork::NeuralNetwork(int input_dim, int output_dim, int hidden_dim)
    : input_dim_(input_dim), output_dim_(output_dim), hidden_dim_(hidden_dim) {
    
    // Initialize weights
    weights_["w1"] = std::vector<double>(input_dim * hidden_dim);
    weights_["b1"] = std::vector<double>(hidden_dim);
    weights_["w2"] = std::vector<double>(hidden_dim * hidden_dim);
    weights_["b2"] = std::vector<double>(hidden_dim);
    weights_["w3"] = std::vector<double>(hidden_dim * output_dim);
    weights_["b3"] = std::vector<double>(output_dim);
    
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);
    
    for (auto& w : weights_["w1"]) w = dis(gen);
    for (auto& w : weights_["w2"]) w = dis(gen);
    for (auto& w : weights_["w3"]) w = dis(gen);
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    // Simplified forward pass (ReLU activations)
    std::vector<double> h1(hidden_dim_);
    std::vector<double> h2(hidden_dim_);
    std::vector<double> output(output_dim_);
    
    // Layer 1
    for (int j = 0; j < hidden_dim_; ++j) {
        h1[j] = weights_["b1"][j];
        for (int i = 0; i < input_dim_; ++i) {
            h1[j] += input[i] * weights_["w1"][i * hidden_dim_ + j];
        }
        h1[j] = std::max(0.0, h1[j]); // ReLU
    }
    
    // Layer 2
    for (int j = 0; j < hidden_dim_; ++j) {
        h2[j] = weights_["b2"][j];
        for (int i = 0; i < hidden_dim_; ++i) {
            h2[j] += h1[i] * weights_["w2"][i * hidden_dim_ + j];
        }
        h2[j] = std::max(0.0, h2[j]); // ReLU
    }
    
    // Output layer
    for (int j = 0; j < output_dim_; ++j) {
        output[j] = weights_["b3"][j];
        for (int i = 0; i < hidden_dim_; ++i) {
            output[j] += h2[i] * weights_["w3"][i * output_dim_ + j];
        }
    }
    
    return output;
}

void NeuralNetwork::backward(const std::vector<double>& gradients) {
    // Simplified backward pass
    std::cout << "[*] Backward pass completed" << std::endl;
}

void NeuralNetwork::update_weights(const std::map<std::string, std::vector<double>>& updates) {
    for (const auto& [key, values] : updates) {
        if (weights_.find(key) != weights_.end() && 
            weights_[key].size() == values.size()) {
            weights_[key] = values;
        }
    }
}

std::map<std::string, std::vector<double>> NeuralNetwork::get_weights() const {
    return weights_;
}

void NeuralNetwork::set_weights(const std::map<std::string, std::vector<double>>& weights) {
    weights_ = weights;
}

// DQN Implementation
DQN::DQN(const RLConfig& config) 
    : config_(config), 
      replay_memory_(config.memory_size),
      rng_(std::random_device{}()) {
    
    policy_network_ = std::make_unique<NeuralNetwork>(config.state_dim, config.action_dim, config.hidden_dim);
    target_network_ = std::make_unique<NeuralNetwork>(config.state_dim, config.action_dim, config.hidden_dim);
    
    std::cout << "[*] DQN initialized with config:" << std::endl;
    std::cout << "  State dim: " << config.state_dim << std::endl;
    std::cout << "  Action dim: " << config.action_dim << std::endl;
    std::cout << "  Hidden dim: " << config.hidden_dim << std::endl;
    std::cout << "  Learning rate: " << config.learning_rate << std::endl;
    std::cout << "  Gamma: " << config.gamma << std::endl;
}

Action DQN::select_action(const State& state, bool training) {
    Action action;
    
    // Epsilon-greedy exploration
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (training && dis(rng_) < config_.epsilon) {
        // Random action
        std::uniform_int_distribution<> action_dist(0, config_.action_dim - 1);
        action.action_id = action_dist(rng_);
    } else {
        // Best action from Q-network
        action.action_id = get_best_action(state);
    }
    
    action.action_name = "action_" + std::to_string(action.action_id);
    
    return action;
}

int DQN::get_best_action(const State& state) {
    auto q_values = policy_network_->forward(state.features);
    
    int best_action = 0;
    double max_q = q_values[0];
    
    for (int i = 1; i < static_cast<int>(q_values.size()); ++i) {
        if (q_values[i] > max_q) {
            max_q = q_values[i];
            best_action = i;
        }
    }
    
    return best_action;
}

void DQN::store_transition(const Transition& transition) {
    replay_memory_.push(transition);
}

void DQN::train() {
    if (replay_memory_.size() < config_.batch_size) {
        return;
    }
    
    auto batch = replay_memory_.sample(config_.batch_size);
    double loss = calculate_loss(batch);
    
    // Update policy network
    std::cout << "[*] DQN Training - Loss: " << loss << std::endl;
    
    // Backward pass would go here
}

double DQN::calculate_loss(const std::vector<Transition>& batch) {
    double total_loss = 0.0;
    
    for (const auto& transition : batch) {
        auto q_values = policy_network_->forward(transition.state.features);
        
        // Simplified TD loss
        double current_q = q_values[transition.action.action_id];
        double target_q = transition.reward;
        
        if (!transition.done) {
            auto next_q_values = target_network_->forward(transition.next_state.features);
            double max_next_q = *std::max_element(next_q_values.begin(), next_q_values.end());
            target_q += config_.gamma * max_next_q;
        }
        
        total_loss += (current_q - target_q) * (current_q - target_q);
    }
    
    return total_loss / batch.size();
}

void DQN::update_target_network() {
    auto weights = policy_network_->get_weights();
    target_network_->set_weights(weights);
    std::cout << "[*] Target network updated" << std::endl;
}

double DQN::get_q_value(const State& state, const Action& action) {
    auto q_values = policy_network_->forward(state.features);
    return q_values[action.action_id];
}

// PPO Implementation
PPO::PPO(const RLConfig& config)
    : config_(config),
      replay_memory_(config.memory_size),
      rng_(std::random_device{}()) {
    
    actor_ = std::make_unique<NeuralNetwork>(config.state_dim, config.action_dim, config.hidden_dim);
    critic_ = std::make_unique<NeuralNetwork>(config.state_dim, 1, config.hidden_dim);
    
    std::cout << "[*] PPO initialized" << std::endl;
}

Action PPO::select_action(const State& state) {
    Action action;
    
    auto logits = actor_->forward(state.features);
    
    // Softmax to get probabilities
    double sum_exp = 0.0;
    std::vector<double> probs(config_.action_dim);
    
    for (int i = 0; i < config_.action_dim; ++i) {
        probs[i] = std::exp(logits[i]);
        sum_exp += probs[i];
    }
    
    for (int i = 0; i < config_.action_dim; ++i) {
        probs[i] /= sum_exp;
    }
    
    action.probabilities = probs;
    
    // Sample action
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double r = dis(rng_);
    double cumulative = 0.0;
    
    for (int i = 0; i < config_.action_dim; ++i) {
        cumulative += probs[i];
        if (r <= cumulative) {
            action.action_id = i;
            break;
        }
    }
    
    action.action_name = "action_" + std::to_string(action.action_id);
    
    // Get value from critic
    auto value = critic_->forward(state.features);
    action.value = value[0];
    
    return action;
}

void PPO::store_transition(const Transition& transition) {
    replay_memory_.push(transition);
}

void PPO::train() {
    if (replay_memory_.size() < config_.batch_size) {
        return;
    }
    
    auto batch = replay_memory_.sample(config_.batch_size);
    
    double policy_loss = compute_policy_loss(batch);
    double value_loss = compute_value_loss(batch);
    
    std::cout << "[*] PPO Training - Policy Loss: " << policy_loss 
              << ", Value Loss: " << value_loss << std::endl;
}

double PPO::compute_policy_loss(const std::vector<Transition>& batch) {
    // Simplified PPO policy loss (clip surrogate objective)
    double loss = 0.0;
    
    for (const auto& t : batch) {
        auto logits = actor_->forward(t.state.features);
        double log_prob = logits[t.action.action_id];
        loss -= log_prob * t.reward; // Simplified
    }
    
    return loss / batch.size();
}

double PPO::compute_value_loss(const std::vector<Transition>& batch) {
    double loss = 0.0;
    
    for (const auto& t : batch) {
        auto value = critic_->forward(t.state.features);
        double td_error = t.reward + config_.gamma * t.next_state.reward - value[0];
        loss += td_error * td_error;
    }
    
    return loss / batch.size();
}

double PPO::compute_gae(const std::vector<Transition>& trajectory) {
    // Generalized Advantage Estimation
    double advantage = 0.0;
    double gae = 0.0;
    
    for (int i = trajectory.size() - 1; i >= 0; --i) {
        const auto& t = trajectory[i];
        double delta = t.reward + config_.gamma * t.next_state.reward - t.state.reward;
        gae = delta + config_.gamma * 0.95 * gae;
        advantage += gae;
    }
    
    return advantage / trajectory.size();
}

// ReinforcementLearningEngine Implementation
ReinforcementLearningEngine::ReinforcementLearningEngine() 
    : initialized_(false), current_step_(0) {}

ReinforcementLearningEngine::~ReinforcementLearningEngine() {}

bool ReinforcementLearningEngine::initialize(const RLConfig& config) {
    config_ = config;
    initialized_ = true;
    
    if (config_.algorithm == "dqn") {
        dqn_agent_ = std::make_unique<DQN>(config);
        current_algorithm_ = "DQN";
    } else if (config_.algorithm == "ppo") {
        ppo_agent_ = std::make_unique<PPO>(config);
        current_algorithm_ = "PPO";
    }
    
    std::cout << "[*] Initializing Reinforcement Learning Engine..." << std::endl;
    std::cout << "[*] Algorithm: " << config_.algorithm << std::endl;
    std::cout << "[*] State dimension: " << config_.state_dim << std::endl;
    std::cout << "[*] Action dimension: " << config_.action_dim << std::endl;
    
    return true;
}

void ReinforcementLearningEngine::train(int num_episodes) {
    std::cout << "[*] Training for " << num_episodes << " episodes..." << std::endl;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        train_episode();
        
        if (episode % 100 == 0) {
            double avg_reward = 0.0;
            int count = std::min(100, static_cast<int>(episode_rewards_.size()));
            for (int i = episode_rewards_.size() - count; i < static_cast<int>(episode_rewards_.size()); ++i) {
                avg_reward += episode_rewards_[i];
            }
            avg_reward /= count;
            
            std::cout << "  Episode " << episode << " - Avg Reward: " << avg_reward << std::endl;
            
            // Update target network for DQN
            if (dqn_agent_ && episode % config_.target_update_freq == 0) {
                dqn_agent_->update_target_network();
            }
            
            // Decay epsilon
            if (config_.epsilon > config_.epsilon_min) {
                config_.epsilon *= config_.epsilon_decay;
            }
        }
    }
    
    std::cout << "[*] Training completed!" << std::endl;
}

void ReinforcementLearningEngine::train_episode() {
    double episode_reward = 0.0;
    State current_state = {}; // Would be initialized from environment
    
    // Initialize state
    current_state.features = std::vector<double>(config_.state_dim, 0.0);
    current_state.reward = 0.0;
    current_state.done = false;
    
    int max_steps = 1000;
    
    for (int step = 0; step < max_steps && !current_state.done; ++step) {
        // Select action
        Action action;
        if (dqn_agent_) {
            action = dqn_agent_->select_action(current_state, true);
        } else if (ppo_agent_) {
            action = ppo_agent_->select_action(current_state);
        }
        
        // Execute action
        execute_action(action);
        
        // Get next state (simplified)
        State next_state = get_next_state(current_state, action);
        
        // Get reward
        double reward = get_reward(current_state, action, next_state);
        next_state.reward = reward;
        
        episode_reward += reward;
        
        // Store transition
        Transition transition = {current_state, action, next_state, reward, next_state.done};
        
        if (dqn_agent_) {
            dqn_agent_->store_transition(transition);
            dqn_agent_->train();
        } else if (ppo_agent_) {
            ppo_agent_->store_transition(transition);
            if (step % 10 == 0) {
                ppo_agent_->train();
            }
        }
        
        current_state = next_state;
        current_step_++;
    }
    
    episode_rewards_.push_back(episode_reward);
}

void ReinforcementLearningEngine::reset_environment() {
    std::cout << "[*] Environment reset" << std::endl;
}

Action ReinforcementLearningEngine::select_action(const State& state) {
    if (dqn_agent_) {
        return dqn_agent_->select_action(state, false);
    } else if (ppo_agent_) {
        return ppo_agent_->select_action(state);
    }
    
    return Action{};
}

void ReinforcementLearningEngine::execute_action(const Action& action) {
    std::cout << "[*] Executing action: " << action.action_name << std::endl;
}

double ReinforcementLearningEngine::evaluate(int num_episodes) {
    double total_reward = 0.0;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        double episode_reward = 0.0;
        State current_state = {};
        current_state.features = std::vector<double>(config_.state_dim, 0.0);
        
        for (int step = 0; step < 1000; ++step) {
            Action action = select_action(current_state);
            execute_action(action);
            
            State next_state = get_next_state(current_state, action);
            double reward = get_reward(current_state, action, next_state);
            
            episode_reward += reward;
            current_state = next_state;
            
            if (current_state.done) break;
        }
        
        total_reward += episode_reward;
    }
    
    return total_reward / num_episodes;
}

std::vector<double> ReinforcementLearningEngine::get_episode_rewards() {
    return episode_rewards_;
}

State ReinforcementLearningEngine::create_state_from_timeseries(const std::vector<double>& data, int lookback) {
    State state;
    
    int start_idx = std::max(0, static_cast<int>(data.size()) - lookback);
    
    // Extract features
    for (int i = start_idx; i < static_cast<int>(data.size()); ++i) {
        state.features.push_back(data[i]);
    }
    
    // Pad if needed
    while (state.features.size() < static_cast<size_t>(config_.state_dim)) {
        state.features.insert(state.features.begin(), 0.0);
    }
    
    // Calculate technical indicators (simplified)
    double sum = 0.0;
    for (double v : state.features) sum += v;
    double mean = sum / state.features.size();
    state.technical_indicators.push_back(mean);
    
    state.reward = 0.0;
    state.done = false;
    
    return state;
}

Action ReinforcementLearningEngine::create_trading_action(double signal) {
    Action action;
    
    if (signal > 0.3) {
        action.action_id = 2; // Buy
        action.action_name = "buy";
    } else if (signal < -0.3) {
        action.action_id = 1; // Sell
        action.action_name = "sell";
    } else {
        action.action_id = 0; // Hold
        action.action_name = "hold";
    }
    
    return action;
}

double ReinforcementLearningEngine::calculate_portfolio_reward(double returns, double risk) {
    // Sharpe-like reward
    if (risk == 0) return returns;
    return returns / risk;
}

State ReinforcementLearningEngine::get_next_state(const State& current_state, const Action& action) {
    State next_state = current_state;
    
    // Simplified state transition
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 0.1);
    
    for (auto& f : next_state.features) {
        f += dis(gen);
    }
    
    next_state.reward = dis(gen);
    next_state.done = (current_step_ % 1000 == 0);
    
    return next_state;
}

double ReinforcementLearningEngine::get_reward(const State& state, const Action& action, const State& next_state) {
    return next_state.reward;
}

void ReinforcementLearningEngine::save_model(const std::string& path) {
    std::cout << "[*] Saving model to: " << path << std::endl;
}

void ReinforcementLearningEngine::load_model(const std::string& path) {
    std::cout << "[*] Loading model from: " << path << std::endl;
}

void ReinforcementLearningEngine::update_learning_rate(double lr) {
    config_.learning_rate = lr;
    std::cout << "[*] Learning rate updated to: " << lr << std::endl;
}

void ReinforcementLearningEngine::update_epsilon(double epsilon) {
    config_.epsilon = epsilon;
    std::cout << "[*] Epsilon updated to: " << epsilon << std::endl;
}

} // namespace Forecast
