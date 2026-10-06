// src/generation/dynamic_temperature.cpp - Adaptive Temperature Controller Implementation
#include "strata/generation/dynamic_temperature.hpp"

#include <cmath>

namespace strata::generation {

DynamicTemperatureController::DynamicTemperatureController(const DynamicTemperatureConfig& config)
    : config_(config) {
    state_.current_temperature = config.base_temperature;
}

DynamicTemperatureController::~DynamicTemperatureController() = default;

void DynamicTemperatureController::reset(double base_temp) {
    config_.base_temperature = base_temp;
    state_.current_temperature = base_temp;
    state_.recovery_attempts = 0;
    state_.tokens_since_adjustment = 0;
    state_.in_recovery = false;
    state_.last_reason.clear();
}

double DynamicTemperatureController::update(const GenerationLoopVerdict& verdict, int32_t token_id) {
    (void)token_id;
    if (!config_.enabled) {
        return config_.base_temperature;
    }

    // Do not alter deterministic generation unless explicitly enabled
    if (config_.base_temperature <= 0.0 && !config_.deterministic_mode) {
        return 0.0;
    }

    state_.tokens_since_adjustment++;

    // Check if recovery intervention is needed
    if (verdict.confidence == LoopConfidence::kSuspicious ||
        verdict.confidence == LoopConfidence::kProbableLoop ||
        verdict.diversity_ratio < config_.min_diversity) {

        if ((!state_.in_recovery || state_.tokens_since_adjustment >= 4) &&
            state_.recovery_attempts < config_.max_recovery_attempts &&
            state_.current_temperature < config_.max_temperature) {

            state_.current_temperature = std::min(
                config_.max_temperature,
                state_.current_temperature + config_.step
            );
            state_.recovery_attempts++;
            state_.tokens_since_adjustment = 0;
            state_.in_recovery = true;
            state_.last_reason = "Elevated temperature due to repetitive pattern / low diversity (" +
                                 verdict.reason + ")";
        }
    } else if (verdict.confidence == LoopConfidence::kNormal && state_.in_recovery) {
        // Hysteresis & cooldown: gradually relax temperature back to baseline
        if (state_.tokens_since_adjustment >= config_.cooldown_tokens) {
            if (state_.current_temperature > config_.base_temperature) {
                state_.current_temperature = std::max(
                    config_.base_temperature,
                    state_.current_temperature - config_.decrease_rate
                );
            }
            if (std::abs(state_.current_temperature - config_.base_temperature) < 1e-4) {
                state_.current_temperature = config_.base_temperature;
                state_.in_recovery = false;
                state_.recovery_attempts = 0;
            }
        }
    }

    return state_.current_temperature;
}

} // namespace strata::generation
