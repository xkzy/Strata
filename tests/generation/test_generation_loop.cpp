// tests/generation/test_generation_loop.cpp - Test Suite for Generation Loop Detector & Dynamic Temperature
#include "strata/generation/dynamic_temperature.hpp"
#include "strata/generation/generation_loop_detector.hpp"

#include <cassert>
#include <iostream>
#include <vector>

void test_single_token_repeat() {
    std::cout << "[Test 1/8] Testing Single Token Repetition Loop..." << std::endl;
    strata::generation::GenerationLoopConfig cfg;
    cfg.max_single_token_repeat = 8;
    strata::generation::GenerationLoopDetector detector(cfg);

    strata::generation::GenerationLoopVerdict v;
    for (int i = 0; i < 7; ++i) {
        v = detector.feed_token(42, "the");
        assert(!v.should_stop);
    }

    v = detector.feed_token(42, "the"); // 8th repeat
    assert(v.should_stop);
    assert(v.confidence == strata::generation::LoopConfidence::kConfirmedLoop);
    assert(v.loop_type == strata::generation::LoopType::kSingleToken);
    assert(v.finish_reason == "generation_loop");
    std::cout << "  Passed. Single-token runaway cleanly stopped." << std::endl;
}

void test_ngram_repeat_loop() {
    std::cout << "[Test 2/8] Testing N-gram Sequence Loop (A B C D)..." << std::endl;
    strata::generation::GenerationLoopConfig cfg;
    cfg.ngram_sizes = {4};
    cfg.max_ngram_repetitions = 3;
    strata::generation::GenerationLoopDetector detector(cfg);

    std::vector<int32_t> pattern = {101, 102, 103, 104};
    strata::generation::GenerationLoopVerdict v;

    for (size_t rep = 0; rep < 2; ++rep) {
        for (int32_t t : pattern) {
            v = detector.feed_token(t, "word");
            assert(!v.should_stop);
        }
    }

    // 3rd repetition: should stop at end of 3rd cycle
    for (size_t i = 0; i < pattern.size() - 1; ++i) {
        v = detector.feed_token(pattern[i], "word");
        assert(!v.should_stop);
    }
    v = detector.feed_token(pattern.back(), "word");
    assert(v.should_stop);
    assert(v.confidence == strata::generation::LoopConfidence::kConfirmedLoop);
    assert(v.period == 4);
    assert(v.repetitions == 3);
    std::cout << "  Passed. 4-gram repetition loop intercepted." << std::endl;
}

void test_periodic_cycle_detection() {
    std::cout << "[Test 3/8] Testing Variable Periodic Cycle Detection..." << std::endl;
    strata::generation::GenerationLoopConfig cfg;
    cfg.max_periodic_period = 16;
    cfg.max_periodic_repetitions = 3;
    strata::generation::GenerationLoopDetector detector(cfg);

    // Period 6 cycle: (1, 2, 3, 4, 5, 6)
    std::vector<int32_t> cycle = {1, 2, 3, 4, 5, 6};
    strata::generation::GenerationLoopVerdict v;

    for (size_t rep = 0; rep < 2; ++rep) {
        for (int32_t t : cycle) {
            v = detector.feed_token(t, "tok");
            assert(!v.should_stop);
        }
    }

    for (size_t i = 0; i < cycle.size() - 1; ++i) {
        v = detector.feed_token(cycle[i], "tok");
        assert(!v.should_stop);
    }
    v = detector.feed_token(cycle.back(), "tok");
    assert(v.should_stop);
    assert(v.period == 6);
    assert(v.repetitions == 3);
    std::cout << "  Passed. Period-6 cyclic loop stopped." << std::endl;
}

void test_structured_code_false_positive_protection() {
    std::cout << "[Test 4/8] Testing Code & Structured Output False-Positive Protection..." << std::endl;
    strata::generation::GenerationLoopConfig cfg;
    cfg.code_protection_enabled = true;
    cfg.structured_output_protection = true;
    strata::generation::GenerationLoopDetector detector(cfg);

    // Simulate repetitive indentation or loop statements in code: "    for (int i = 0; i < n; ++i) {"
    std::vector<std::pair<int32_t, std::string>> code_tokens = {
        {10, "    "}, {11, "for"}, {12, " "}, {13, "("}, {14, "int"}, {15, " "}, {16, "i"},
        {17, " "}, {18, "="}, {19, " "}, {20, "0"}, {21, ";"}, {22, " "}, {23, "i"},
        {24, " "}, {25, "<"}, {26, " "}, {27, "n"}, {28, ";"}, {29, " "}, {30, "++"},
        {31, "i"}, {32, ")"}, {33, " "}, {34, "{"}, {35, "\n"},
        {10, "    "}, {10, "    "}, {36, "return"}, {37, " "}, {38, "i"}, {39, ";"}, {40, "\n"},
        {10, "    "}, {41, "}"}
    };

    for (const auto& [id, text] : code_tokens) {
        auto v = detector.feed_token(id, text);
        assert(!v.should_stop && "Legitimate code must not trigger premature loop stop");
    }
    std::cout << "  Passed. Structured code output was preserved without false positives." << std::endl;
}

void test_diversity_entropy_tracking() {
    std::cout << "[Test 5/8] Testing Diversity and Shannon Entropy Tracking..." << std::endl;
    strata::generation::GenerationLoopConfig cfg;
    strata::generation::GenerationLoopDetector detector(cfg);

    // Diverse stream
    for (int i = 0; i < 20; ++i) {
        detector.feed_token(1000 + i, "unique");
    }
    assert(detector.current_diversity() > 0.9);
    assert(detector.current_entropy() > 0.9);

    std::cout << "  Passed. Diversity & entropy correctly computed." << std::endl;
}

void test_dynamic_temperature_escalation_and_cooldown() {
    std::cout << "[Test 6/8] Testing Dynamic Temperature Escalation & Recovery..." << std::endl;
    strata::generation::DynamicTemperatureConfig cfg;
    cfg.enabled = true;
    cfg.base_temperature = 0.7;
    cfg.max_temperature = 1.1;
    cfg.step = 0.2;
    cfg.cooldown_tokens = 5;
    cfg.decrease_rate = 0.05;

    strata::generation::DynamicTemperatureController controller(cfg);
    assert(controller.current_temperature() == 0.7);

    // Suspicious verdict -> should escalate temperature
    strata::generation::GenerationLoopVerdict v_suspicious;
    v_suspicious.confidence = strata::generation::LoopConfidence::kSuspicious;
    v_suspicious.diversity_ratio = 0.2;
    v_suspicious.reason = "Repetition detected";

    double temp1 = controller.update(v_suspicious, 100);
    (void)temp1;
    assert(temp1 > 0.7);
    assert(controller.state().in_recovery);

    // Once recovered (normal stream), temperature should gradually relax to baseline
    strata::generation::GenerationLoopVerdict v_normal;
    v_normal.confidence = strata::generation::LoopConfidence::kNormal;
    v_normal.diversity_ratio = 0.8;

    for (int i = 0; i < 15; ++i) {
        controller.update(v_normal, 200 + i);
    }

    assert(controller.current_temperature() == 0.7);
    assert(!controller.state().in_recovery);
    std::cout << "  Passed. Adaptive temperature escalated and cooled down gracefully." << std::endl;
}

void test_deterministic_mode_immutability() {
    std::cout << "[Test 7/8] Testing Deterministic Mode (temp=0) Immutability..." << std::endl;
    strata::generation::DynamicTemperatureConfig cfg;
    cfg.enabled = true;
    cfg.base_temperature = 0.0;
    cfg.deterministic_mode = false; // Default: strict deterministic preservation

    strata::generation::DynamicTemperatureController controller(cfg);

    strata::generation::GenerationLoopVerdict v_suspicious;
    v_suspicious.confidence = strata::generation::LoopConfidence::kSuspicious;
    v_suspicious.diversity_ratio = 0.1;

    double temp = controller.update(v_suspicious, 100);
    (void)temp;
    assert(temp == 0.0 && "Deterministic mode must remain at 0.0 when deterministic_mode is false");
    std::cout << "  Passed. Deterministic temperature 0.0 strictly preserved." << std::endl;
}

void test_reset_isolation() {
    std::cout << "[Test 8/8] Testing State Isolation & Reset..." << std::endl;
    strata::generation::GenerationLoopDetector detector;
    for (int i = 0; i < 10; ++i) {
        detector.feed_token(999, "repeat");
    }
    detector.reset();
    assert(detector.total_tokens_seen() == 0);
    assert(detector.current_diversity() == 1.0);

    std::cout << "  Passed. Detector reset cleanly for subsequent requests." << std::endl;
}

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   RUNNING STRATA GENERATION LOOP & DYNAMIC TEMP TEST SUITE     " << std::endl;
    std::cout << "=================================================================" << std::endl;

    test_single_token_repeat();
    test_ngram_repeat_loop();
    test_periodic_cycle_detection();
    test_structured_code_false_positive_protection();
    test_diversity_entropy_tracking();
    test_dynamic_temperature_escalation_and_cooldown();
    test_deterministic_mode_immutability();
    test_reset_isolation();

    std::cout << "=================================================================" << std::endl;
    std::cout << "   ALL GENERATION LOOP TESTS PASSED (8/8)                        " << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
