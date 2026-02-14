#ifndef REINFORCEMENT_LEARNING_H
#define REINFORCEMENT_LEARNING_H

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <random>

namespace Forecast {

// Reinforcement Learning for time series trading and control
struct RLConfig {
    int state_dim;
    int action_dim;
    int hidden_dim;
    double learning_rate;
    double gamma; // discount factor
    double epsilon; // exploration rate
    double epsilon_decay;
    double epsilon_min;
    int memory_size;
    int batch_size;
    int target_update_freq;
    std::string algorithm; // "dqn", "ppo", "a2c", "ddpg"
    bool enable_double_dqn;
    bool enable_dueling;
    bool enable_prioritized_replay;
};

struct State {
    std::vector<double> features;
    std::vector<double> technical_indicators;
    std::vector<double> sentiment_scores;
    double reward;
    bool done;
};

struct Action {
    int action_id;
    std::string action_name;
    std::vector<double> probabilities;
    double value;
};

struct Transition {
    State state;
    Action action;
    State next_state;
    double reward;
    bool done;
};

class ReplayMemory {
public:
    ReplayMemory(int capacity);
    void push(const Transition& transition);
    std::vector<Transition> sample(int batch_size);
    size_t size() const;
    bool is_full() const;

private:
    int capacity_;
    std::vector<Transition> memory_;
    int position_;
};

class NeuralNetwork {
public:
    NeuralNetwork(int input_dim, int output_dim, int hidden_dim);
    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& gradients);
    void update_weights(const std::map<std::string, std::vector<double>>& updates);
    std::map<std::string, std::vector<double>> get_weights() const;
    void set_weights(const std::map<std::string, std::vector<double>>& weights);

private:
    int input_dim_, output_dim_, hidden_dim_;
    std::map<std::string, std::vector<double>> weights_;
};

class DQN {
public:
    DQN(const RLConfig& config);
    Action select_action(const State& state, bool training = true);
    void store_transition(const Transition& transition);
    void train();
    void update_target_network();
    double get_q_value(const State& state, const Action& action);

private:
    RLConfig config_;
    std::unique_ptr<NeuralNetwork> policy_network_;
    std::unique_ptr<NeuralNetwork> target_network_;
    ReplayMemory replay_memory_;
    std::mt19937 rng_;
    
    double calculate_loss(const std::vector<Transition>& batch);
    int get_best_action(const State& state);
};

class PPO {
public:
    PPO(const RLConfig& config);
    Action select_action(const State& state);
    void store_transition(const Transition& transition);
    void train();
    double compute_gae(const std::vector<Transition>& trajectory);

private:
    RLConfig config_;
    std::unique_ptr<NeuralNetwork> actor_;
    std::unique_ptr<NeuralNetwork> critic_;
    ReplayMemory replay_memory_;
    std::mt19937 rng_;
    
    double compute_policy_loss(const std::vector<Transition>& batch);
    double compute_value_loss(const std::vector<Transition>& batch);
};

class ReinforcementLearningEngine {
public:
    ReinforcementLearningEngine();
    ~ReinforcementLearningEngine();

    bool initialize(const RLConfig& config);
    
    // Training
    void train(int num_episodes);
    void train_episode();
    void reset_environment();
    
    // Actions
    Action select_action(const State& state);
    void execute_action(const Action& action);
    
    // Evaluation
    double evaluate(int num_episodes);
    std::vector<double> get_episode_rewards();
    
    // Time Series Specific
    State create_state_from_timeseries(const std::vector<double>& data, int lookback);
    Action create_trading_action(double signal);
    double calculate_portfolio_reward(double returns, double risk);
    
    // Save/Load
    void save_model(const std::string& path);
    void load_model(const std::string& path);
    
    // Hyperparameters
    void update_learning_rate(double lr);
    void update_epsilon(double epsilon);

private:
    bool initialized_;
    RLConfig config_;
    std::unique_ptr<DQN> dqn_agent_;
    std::unique_ptr<PPO> ppo_agent_;
    std::string current_algorithm_;
    std::vector<double> episode_rewards_;
    int current_step_;
    
    State get_next_state(const State& current_state, const Action& action);
    double get_reward(const State& state, const Action& action, const State& next_state);
};

} // namespace Forecast

#endif // REINFORCEMENT_LEARNING_H
