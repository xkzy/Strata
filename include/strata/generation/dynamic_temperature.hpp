// include/strata/generation/dynamic_temperature.hpp - Adaptive Temperature Controller
#pragma once

#include "strata/generation/generation_loop_detector.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

namespace strata::generation {

struct DynamicTemperatureConfig {
    bool enabled = false;
    double base_temperature = 0.7;
    double max_temperature = 1.2;
    double step = 0.15;
    double increase_threshold = 0.55; // Repetition/low diversity threshold
    double decrease_rate = 0.02;      // Rate of return toward base temperature per token
    size_t cooldown_tokens = 16;
    double min_diversity = 0.35;
    size_t max_recovery_attempts = 3;
    bool deterministic_mode = false;  // Allow recovery if base_temperature == 0.0
};

struct DynamicTemperatureState {
    double current_temperature = 0.7;
    size_t recovery_attempts = 0;
    size_t tokens_since_adjustment = 0;
    bool in_recovery = false;
    std::string last_reason;
};

class DynamicTemperatureController {
public:
    explicit DynamicTemperatureController(const DynamicTemperatureConfig& config = DynamicTemperatureConfig());
    ~DynamicTemperatureController();

    // Updates temperature based on online generation metrics. Returns current effective temperature.
    double update(const GenerationLoopVerdict& verdict, int32_t token_id);

    // Reset state for new request / session
    void reset(double base_temp = 0.7);

    // Current effective temperature
    double current_temperature() const { return state_.current_temperature; }
    const DynamicTemperatureState& state() const { return state_; }
    const DynamicTemperatureConfig& config() const { return config_; }
    void set_config(const DynamicTemperatureConfig& config) { config_ = config; }

private:
    DynamicTemperatureConfig config_;
    DynamicTemperatureState state_;
};

} // namespace strata::generation
