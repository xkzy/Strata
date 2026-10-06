# serve/test_generation_loop.py - Unit tests for generation loop detector & dynamic temp
import unittest
from serve.generation_loop import (
    GenerationLoopConfig,
    GenerationLoopDetector,
    LoopConfidence,
    LoopType,
    DynamicTemperatureConfig,
    DynamicTemperatureController,
)


class TestGenerationLoop(unittest.TestCase):
    def test_single_token_repeat(self):
        cfg = GenerationLoopConfig(max_single_token_repeat=6)
        detector = GenerationLoopDetector(cfg)

        for _ in range(5):
            v = detector.feed_token(10, "repeat")
            self.assertFalse(v.should_stop)

        v = detector.feed_token(10, "repeat")
        self.assertTrue(v.should_stop)
        self.assertEqual(v.confidence, LoopConfidence.CONFIRMED_LOOP)
        self.assertEqual(v.loop_type, LoopType.SINGLE_TOKEN)
        self.assertEqual(v.finish_reason, "generation_loop")

    def test_ngram_repeat_loop(self):
        cfg = GenerationLoopConfig(ngram_sizes=[3], max_ngram_repetitions=3)
        detector = GenerationLoopDetector(cfg)

        pattern = [1, 2, 3]
        for _ in range(2):
            for t in pattern:
                v = detector.feed_token(t, "w")
                self.assertFalse(v.should_stop)

        for t in pattern[:-1]:
            v = detector.feed_token(t, "w")
            self.assertFalse(v.should_stop)

        v = detector.feed_token(pattern[-1], "w")
        self.assertTrue(v.should_stop)
        self.assertEqual(v.period, 3)
        self.assertEqual(v.repetitions, 3)

    def test_periodic_cycle_detection(self):
        cfg = GenerationLoopConfig(max_periodic_period=8, max_periodic_repetitions=3)
        detector = GenerationLoopDetector(cfg)

        cycle = [10, 20, 30, 40, 50]
        for _ in range(2):
            for t in cycle:
                v = detector.feed_token(t, "tok")
                self.assertFalse(v.should_stop)

        for t in cycle[:-1]:
            v = detector.feed_token(t, "tok")
            self.assertFalse(v.should_stop)

        v = detector.feed_token(cycle[-1], "tok")
        self.assertTrue(v.should_stop)
        self.assertEqual(v.period, 5)
        self.assertEqual(v.repetitions, 3)

    def test_structured_code_protection(self):
        cfg = GenerationLoopConfig(code_protection_enabled=True, structured_output_protection=True)
        detector = GenerationLoopDetector(cfg)

        # Code statement with indentation and braces
        tokens = [
            (1, "    "), (2, "for"), (3, " "), (4, "("), (5, "int"), (6, " "), (7, "i"),
            (8, " "), (9, "="), (10, " "), (11, "0"), (12, ";"), (13, " "), (14, "i"),
            (15, " "), (16, "<"), (17, " "), (18, "n"), (19, ";"), (20, " "), (21, "++"),
            (22, "i"), (23, ")"), (24, " "), (25, "{"), (26, "\n"),
            (1, "    "), (1, "    "), (27, "return"), (28, " "), (29, "i"), (30, ";"), (31, "\n"),
            (1, "    "), (32, "}")
        ]

        for tid, text in tokens:
            v = detector.feed_token(tid, text)
            self.assertFalse(v.should_stop)

    def test_diversity_and_entropy(self):
        cfg = GenerationLoopConfig()
        detector = GenerationLoopDetector(cfg)

        for i in range(20):
            detector.feed_token(100 + i, "token")

        self.assertGreater(detector.current_diversity, 0.9)
        self.assertGreater(detector.current_entropy, 0.9)

    def test_dynamic_temperature_recovery(self):
        cfg = DynamicTemperatureConfig(
            enabled=True,
            base_temperature=0.7,
            max_temperature=1.1,
            step=0.2,
            cooldown_tokens=4,
            decrease_rate=0.05
        )
        ctrl = DynamicTemperatureController(cfg)
        self.assertEqual(ctrl.current_temperature, 0.7)

        v_susp = detector_verdict = GenerationLoopDetector().feed_token(1, "x")
        v_susp.confidence = LoopConfidence.SUSPICIOUS
        v_susp.diversity_ratio = 0.2

        t1 = ctrl.update(v_susp, 1)
        self.assertGreater(t1, 0.7)
        self.assertTrue(ctrl.state.in_recovery)

        v_norm = GenerationLoopDetector().feed_token(2, "y")
        v_norm.confidence = LoopConfidence.NORMAL
        v_norm.diversity_ratio = 0.9

        for _ in range(12):
            ctrl.update(v_norm, 10)

        self.assertAlmostEqual(ctrl.current_temperature, 0.7, places=3)
        self.assertFalse(ctrl.state.in_recovery)


if __name__ == "__main__":
    unittest.main()
