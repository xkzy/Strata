// include/strata/runtime/in_flight_batcher.hpp
#pragma once

#include "strata/kernels/fast_paged_kv.hpp"
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::runtime {

enum class RequestPhase {
    kPending = 0,   // Waiting in queue for admission
    kPrefill = 1,   // Processing prompt tokens in chunks
    kDecode = 2,    // Generating tokens iteration by iteration
    kFinished = 3   // Generation completed (EOS or token limit)
};

struct GenerationRequest {
    uint64_t request_id = 0;
    std::vector<int32_t> prompt_tokens;
    size_t max_new_tokens = 128;
    int32_t eos_token_id = 151643;
    std::vector<int32_t> stop_tokens;

    // Execution state
    RequestPhase phase = RequestPhase::kPending;
    size_t prompt_tokens_processed = 0;
    std::vector<int32_t> generated_tokens;
    uint64_t submit_time_us = 0;
    uint64_t first_token_time_us = 0;
    uint64_t completion_time_us = 0;
    bool is_cancelled = false;
};

struct InFlightBatcherConfig {
    size_t max_batch_size = 16;         // Maximum concurrent requests in active batch
    size_t prefill_chunk_size = 256;    // Chunked prefill token limit per iteration
    bool enable_speculative_decode = false;
    size_t speculative_draft_k = 4;
};

struct IterationStepResult {
    uint64_t iteration_id = 0;
    size_t num_prefill_requests = 0;
    size_t num_decode_requests = 0;
    size_t prefill_tokens_processed = 0;
    std::vector<std::pair<uint64_t, int32_t>> newly_generated_tokens; // (request_id, token_id)
    std::vector<uint64_t> completed_request_ids;
    double iteration_elapsed_ms = 0.0;
};

struct BatcherMetrics {
    size_t pending_requests = 0;
    size_t active_requests = 0;
    size_t in_flight_prefill = 0;
    size_t in_flight_decode = 0;
    size_t total_completed_requests = 0;
    size_t total_iterations_executed = 0;
    size_t total_tokens_generated = 0;
    double avg_ttft_ms = 0.0; // Average Time-to-First-Token in milliseconds
    double avg_itl_ms = 0.0;  // Average Inter-Token-Latency in milliseconds
};

/// NVIDIA-style In-Flight Continuous Batching Scheduler with Chunked Prefill
/// Dynamically evicts completed requests and admits new requests at iteration granularity,
/// interleaving prompt prefill chunks with token decoding steps.
class InFlightBatcher {
public:
    explicit InFlightBatcher(
        std::shared_ptr<strata::kernels::FastPagedKVCache> paged_cache,
        const InFlightBatcherConfig& config = InFlightBatcherConfig())
        : paged_cache_(std::move(paged_cache)), config_(config) {
        if (config_.max_batch_size == 0) config_.max_batch_size = 8;
        if (config_.prefill_chunk_size == 0) config_.prefill_chunk_size = 128;
    }

    ~InFlightBatcher() = default;

    /// Submits a new generation request into the pending admission queue
    bool submit_request(const GenerationRequest& req) {
        std::lock_guard<std::mutex> lock(mu_);
        GenerationRequest r = req;
        r.phase = RequestPhase::kPending;
        r.prompt_tokens_processed = 0;
        r.generated_tokens.clear();
        r.submit_time_us = now_us();
        pending_queue_.push_back(r);
        return true;
    }

    /// Cancels an in-flight or pending request by ID
    bool cancel_request(uint64_t request_id) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& req : pending_queue_) {
            if (req.request_id == request_id) {
                req.is_cancelled = true;
                return true;
            }
        }
        auto it = active_requests_.find(request_id);
        if (it != active_requests_.end()) {
            it->second.is_cancelled = true;
            return true;
        }
        return false;
    }

    /// Executes one continuous batching iteration step.
    /// In a full production loop, token_predictor is called to compute the next token(s).
    /// If token_predictor is nullptr, a mock generator is used for testing and benchmarking.
    IterationStepResult step(
        std::function<int32_t(uint64_t req_id, const std::vector<int32_t>& context)> token_predictor = nullptr) {
        auto start_time = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mu_);

        IterationStepResult result;
        result.iteration_id = ++total_iterations_;

        // 1. Evict completed or cancelled requests from active batch
        std::vector<uint64_t> to_evict;
        for (auto& [req_id, req] : active_requests_) {
            if (req.phase == RequestPhase::kFinished || req.is_cancelled) {
                to_evict.push_back(req_id);
            }
        }

        for (uint64_t req_id : to_evict) {
            auto it = active_requests_.find(req_id);
            if (it != active_requests_.end()) {
                it->second.completion_time_us = now_us();
                completed_requests_.push_back(it->second);
                result.completed_request_ids.push_back(req_id);

                // Release physical blocks in paged KV cache
                if (paged_cache_) {
                    paged_cache_->destroy_sequence(req_id);
                }

                // Update metrics
                total_tokens_generated_ += it->second.generated_tokens.size();
                active_requests_.erase(it);
            }
        }

        // 2. Admit pending requests into active batch if slots and KV cache blocks are available
        while (active_requests_.size() < config_.max_batch_size && !pending_queue_.empty()) {
            GenerationRequest next_req = pending_queue_.front();
            pending_queue_.pop_front();

            if (next_req.is_cancelled) continue;

            // Initialize sequence in Paged KV Cache
            if (paged_cache_) {
                if (!paged_cache_->create_sequence(next_req.request_id)) {
                    // Cache might be out of resources, put back to pending
                    pending_queue_.push_front(next_req);
                    break;
                }
            }

            next_req.phase = RequestPhase::kPrefill;
            active_requests_.emplace(next_req.request_id, std::move(next_req));
        }

        // 3. Process Chunked Prefill & Decode for all active requests
        for (auto& [req_id, req] : active_requests_) {
            if (req.phase == RequestPhase::kPrefill) {
                result.num_prefill_requests++;
                size_t remaining_prompt = req.prompt_tokens.size() - req.prompt_tokens_processed;
                size_t chunk = std::min(remaining_prompt, config_.prefill_chunk_size);

                req.prompt_tokens_processed += chunk;
                result.prefill_tokens_processed += chunk;

                // Prompt prefill complete: transition to Decode phase
                if (req.prompt_tokens_processed >= req.prompt_tokens.size()) {
                    req.phase = RequestPhase::kDecode;
                    req.first_token_time_us = now_us();
                }
            } else if (req.phase == RequestPhase::kDecode) {
                result.num_decode_requests++;

                // Predict next token
                int32_t next_tok = 0;
                if (token_predictor) {
                    std::vector<int32_t> ctx = req.prompt_tokens;
                    ctx.insert(ctx.end(), req.generated_tokens.begin(), req.generated_tokens.end());
                    next_tok = token_predictor(req_id, ctx);
                } else {
                    // Default test token generator
                    next_tok = static_cast<int32_t>(req.generated_tokens.size() + 100);
                }

                req.generated_tokens.push_back(next_tok);
                result.newly_generated_tokens.emplace_back(req_id, next_tok);

                // Check termination conditions
                bool hit_eos = (next_tok == req.eos_token_id);
                bool hit_stop = false;
                for (int32_t stop_tok : req.stop_tokens) {
                    if (next_tok == stop_tok) {
                        hit_stop = true;
                        break;
                    }
                }
                bool hit_max = (req.generated_tokens.size() >= req.max_new_tokens);

                if (hit_eos || hit_stop || hit_max) {
                    req.phase = RequestPhase::kFinished;
                }
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        result.iteration_elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_iteration_ms_ += result.iteration_elapsed_ms;

        return result;
    }

    /// Query current live metrics
    BatcherMetrics metrics() const {
        std::lock_guard<std::mutex> lock(mu_);
        BatcherMetrics m;
        m.pending_requests = pending_queue_.size();
        m.active_requests = active_requests_.size();
        m.total_completed_requests = completed_requests_.size();
        m.total_iterations_executed = total_iterations_;
        m.total_tokens_generated = total_tokens_generated_;

        size_t pf = 0, dec = 0;
        for (const auto& [_, req] : active_requests_) {
            if (req.phase == RequestPhase::kPrefill) pf++;
            else if (req.phase == RequestPhase::kDecode) dec++;
        }
        m.in_flight_prefill = pf;
        m.in_flight_decode = dec;

        // Calculate average TTFT and ITL
        double ttft_sum = 0.0;
        size_t ttft_count = 0;
        for (const auto& req : completed_requests_) {
            if (req.first_token_time_us >= req.submit_time_us && req.submit_time_us > 0) {
                ttft_sum += (req.first_token_time_us - req.submit_time_us) / 1000.0;
                ttft_count++;
            }
        }
        m.avg_ttft_ms = (ttft_count > 0) ? (ttft_sum / ttft_count) : 0.0;

        if (total_tokens_generated_ > 0 && total_iteration_ms_ > 0.0) {
            m.avg_itl_ms = total_iteration_ms_ / static_cast<double>(total_tokens_generated_);
        }

        return m;
    }

    /// Checks if there is any active or pending work remaining
    bool has_active_or_pending_work() const {
        std::lock_guard<std::mutex> lock(mu_);
        return !pending_queue_.empty() || !active_requests_.empty();
    }

    /// Retrieve completed request details by ID
    bool get_completed_request(uint64_t request_id, GenerationRequest& out_req) const {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto& req : completed_requests_) {
            if (req.request_id == request_id) {
                out_req = req;
                return true;
            }
        }
        return false;
    }

private:
    static inline uint64_t now_us() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    std::shared_ptr<strata::kernels::FastPagedKVCache> paged_cache_;
    InFlightBatcherConfig config_;
    mutable std::mutex mu_;

    std::deque<GenerationRequest> pending_queue_;
    std::unordered_map<uint64_t, GenerationRequest> active_requests_;
    std::vector<GenerationRequest> completed_requests_;

    size_t total_iterations_ = 0;
    size_t total_tokens_generated_ = 0;
    double total_iteration_ms_ = 0.0;
};

} // namespace strata::runtime
