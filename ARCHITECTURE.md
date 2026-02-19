# Architecture

## Complexity tier

**Tier 2 (modular C++ prototype)**

## Components
- `time_series_forecaster.cpp`: orchestration entry point for prediction pipeline.
- `time_series_features.cpp`: deterministic feature extraction stage.
- `anomaly_detection_engine.cpp`: anomaly scoring path.
- `federated_learning_engine.cpp`: distributed update logic scaffold.
- `quantum_inspired_optimizer.cpp`: optimization heuristics.
- `reinforcement_learning.cpp`: adaptive policy update scaffold.

## Quality gates
- CI performs CMake configure + parallel build on every push/PR.
