# serve/generation_loop.py - Online Generation Loop Detector & Dynamic Temperature Controller
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class LoopConfidence(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    PROBABLE_LOOP = "PROBABLE_LOOP"
    CONFIRMED_LOOP = "CONFIRMED_LOOP"


class LoopType(str, Enum):
    NONE = "NONE"
    SINGLE_TOKEN = "SINGLE_TOKEN"
    NGRAM = "NGRAM"
    REPEATING_SPAN = "REPEATING_SPAN"
    PERIODIC_CYCLE = "PERIODIC_CYCLE"
    DIVERSITY_COLLAPSE = "DIVERSITY_COLLAPSE"


@dataclass
class GenerationLoopConfig:
    max_single_token_repeat: int = 32
    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3, 4, 8, 16])
    max_ngram_repetitions: int = 5
    max_span_repetitions: int = 4
    min_span_length: int = 12
    window_size: int = 128
    entropy_threshold: float = 0.20
    min_diversity_ratio: float = 0.12
    code_protection_enabled: bool = True
    structured_output_protection: bool = True
    max_periodic_period: int = 32
    max_periodic_repetitions: int = 4


@dataclass
class GenerationLoopVerdict:
    should_stop: bool = False
    confidence: LoopConfidence = LoopConfidence.NORMAL
    loop_type: LoopType = LoopType.NONE
    period: int = 0
    repetitions: int = 0
    diversity_ratio: float = 1.0
    entropy: float = 1.0
    trim_token_count: int = 0
    reason: str = ""
    finish_reason: str = ""


class GenerationLoopDetector:
    """Online, per-token sliding-window generation loop detector."""

    def __init__(self, config: Optional[GenerationLoopConfig] = None):
        self.config = config or GenerationLoopConfig()
        self.token_window: deque = deque()
        self.text_window: deque = deque()
        self.token_counts: Dict[int, int] = defaultdict(int)

        self.last_token: int = -1
        self.single_token_run: int = 0
        self.code_like_tokens_in_window: int = 0

        self.current_diversity: float = 1.0
        self.current_entropy: float = 1.0
        self.total_tokens_seen: int = 0
        self.tokens_saved: int = 0
        self.consecutive_suspicious: int = 0

    @staticmethod
    def is_code_or_structured_token(text: str) -> bool:
        if not text:
            return False
        if "    " in text or "\t" in text:
            return True
        for c in text:
            if c in ("{", "}", "[", "]", "|", ";", ","):
                return True
        if text.strip() in ("for", "while", "def", "function", "return", "const", "let", "var", "import", "from", "class"):
            return True
        return False

    def _hash_ngram(self, start_idx: int, n: int) -> int:
        h = 14695981039346656037
        for i in range(n):
            if start_idx + i < len(self.token_window):
                tok = self.token_window[start_idx + i]
                h = ((h ^ (tok & 0xFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    def _update_diversity_and_entropy(self):
        total = len(self.token_window)
        if total == 0:
            self.current_diversity = 1.0
            self.current_entropy = 1.0
            return

        unique = len(self.token_counts)
        self.current_diversity = unique / total

        entropy = 0.0
        for count in self.token_counts.values():
            p = count / total
            if p > 0.0:
                entropy -= p * math.log2(p)

        max_ent = math.log2(min(total, unique)) if total > 1 and unique > 1 else 1.0
        self.current_entropy = min(1.0, entropy / max_ent) if max_ent > 0.0 else 0.0

    def feed_token(self, token_id: int, token_text: str = "") -> GenerationLoopVerdict:
        self.total_tokens_seen += 1
        verdict = GenerationLoopVerdict()

        # 1. Single Token Repetition
        if token_id == self.last_token:
            self.single_token_run += 1
        else:
            self.last_token = token_id
            self.single_token_run = 1

        max_single = self.config.max_single_token_repeat
        if self.config.structured_output_protection and token_text in (" ", "  ", "    ", "\t", "\n"):
            max_single *= 4

        if self.single_token_run >= max_single:
            verdict.should_stop = True
            verdict.confidence = LoopConfidence.CONFIRMED_LOOP
            verdict.loop_type = LoopType.SINGLE_TOKEN
            verdict.period = 1
            verdict.repetitions = self.single_token_run
            verdict.trim_token_count = self.single_token_run - 1
            verdict.reason = f"Single token repeat loop: token {token_id} repeated {self.single_token_run} times continuously."
            verdict.finish_reason = "generation_loop"
            self.tokens_saved += self.single_token_run
            return verdict

        # Add to window
        self.token_window.append(token_id)
        self.text_window.append(token_text)
        self.token_counts[token_id] += 1

        if self.is_code_or_structured_token(token_text):
            self.code_like_tokens_in_window += 1

        if len(self.token_window) > self.config.window_size:
            old_tok = self.token_window.popleft()
            old_text = self.text_window.popleft()

            if self.token_counts[old_tok] <= 1:
                del self.token_counts[old_tok]
            else:
                self.token_counts[old_tok] -= 1

            if self.is_code_or_structured_token(old_text) and self.code_like_tokens_in_window > 0:
                self.code_like_tokens_in_window -= 1

        self._update_diversity_and_entropy()
        verdict.diversity_ratio = self.current_diversity
        verdict.entropy = self.current_entropy

        n_tokens = len(self.token_window)
        if n_tokens < 4:
            return verdict

        code_ratio = self.code_like_tokens_in_window / n_tokens
        in_structured_mode = self.config.code_protection_enabled and (code_ratio > 0.35)

        # 2. N-gram and Periodic Cycle Checking (period p from 2 up to max_periodic_period)
        for p in range(2, min(self.config.max_periodic_period + 1, n_tokens // 2 + 1)):
            reps = 1
            match = True

            for rep in range(1, n_tokens // p):
                for i in range(p):
                    curr_idx = n_tokens - 1 - i
                    prev_idx = n_tokens - 1 - i - (rep * p)
                    if self.token_window[curr_idx] != self.token_window[prev_idx]:
                        match = False
                        break
                if match:
                    reps += 1
                else:
                    break

            is_ngram = (p <= 4 or p in self.config.ngram_sizes)
            base_limit = self.config.max_ngram_repetitions if is_ngram else self.config.max_periodic_repetitions
            thresh = (base_limit + 2) if in_structured_mode else base_limit
            if reps >= thresh:
                verdict.should_stop = True
                verdict.confidence = LoopConfidence.CONFIRMED_LOOP
                verdict.loop_type = LoopType.NGRAM if p <= 4 else LoopType.PERIODIC_CYCLE
                verdict.period = p
                verdict.repetitions = reps
                verdict.trim_token_count = p * (reps - 1)
                verdict.reason = f"Periodic generation loop detected (period {p} repeated {reps} times)."
                verdict.finish_reason = "generation_loop"
                self.tokens_saved += (p * reps)
                return verdict
            elif reps >= 2:
                verdict.confidence = LoopConfidence.SUSPICIOUS
                verdict.period = p
                verdict.repetitions = reps

        # 3. Repeating Spans via Rolling Hash
        for span_len in self.config.ngram_sizes:
            if span_len >= self.config.min_span_length and n_tokens >= span_len * 2:
                curr_h = self._hash_ngram(n_tokens - span_len, span_len)
                span_reps = 1
                for k in range(1, n_tokens // span_len):
                    prev_h = self._hash_ngram(n_tokens - (k + 1) * span_len, span_len)
                    if curr_h == prev_h:
                        span_reps += 1
                    else:
                        break

                span_thresh = (self.config.max_span_repetitions + 1) if in_structured_mode else self.config.max_span_repetitions
                if span_reps >= span_thresh:
                    verdict.should_stop = True
                    verdict.confidence = LoopConfidence.CONFIRMED_LOOP
                    verdict.loop_type = LoopType.REPEATING_SPAN
                    verdict.period = span_len
                    verdict.repetitions = span_reps
                    verdict.trim_token_count = span_len * (span_reps - 1)
                    verdict.reason = f"Repeating text span loop detected (length {span_len} tokens repeated {span_reps} times)."
                    verdict.finish_reason = "generation_loop"
                    self.tokens_saved += (span_len * span_reps)
                    return verdict

        # 4. Diversity Collapse
        if n_tokens >= 32 and not in_structured_mode:
            if self.current_diversity < self.config.min_diversity_ratio and self.current_entropy < self.config.entropy_threshold:
                self.consecutive_suspicious += 1
                if self.consecutive_suspicious >= 12:
                    verdict.should_stop = True
                    verdict.confidence = LoopConfidence.CONFIRMED_LOOP
                    verdict.loop_type = LoopType.DIVERSITY_COLLAPSE
                    verdict.reason = "Diversity and entropy collapsed in generation stream."
                    verdict.finish_reason = "generation_loop"
                    return verdict
                else:
                    verdict.confidence = LoopConfidence.PROBABLE_LOOP
            else:
                if self.consecutive_suspicious > 0:
                    self.consecutive_suspicious -= 1

        return verdict

    def reset(self):
        self.token_window.clear()
        self.text_window.clear()
        self.token_counts.clear()
        self.last_token = -1
        self.single_token_run = 0
        self.code_like_tokens_in_window = 0
        self.current_diversity = 1.0
        self.current_entropy = 1.0
        self.total_tokens_seen = 0
        self.tokens_saved = 0
        self.consecutive_suspicious = 0


@dataclass
class DynamicTemperatureConfig:
    enabled: bool = False
    base_temperature: float = 0.7
    max_temperature: float = 1.2
    step: float = 0.15
    increase_threshold: float = 0.55
    decrease_rate: float = 0.02
    cooldown_tokens: int = 16
    min_diversity: float = 0.35
    max_recovery_attempts: int = 3
    deterministic_mode: bool = False


@dataclass
class DynamicTemperatureState:
    current_temperature: float = 0.7
    recovery_attempts: int = 0
    tokens_since_adjustment: int = 0
    in_recovery: bool = False
    last_reason: str = ""


class DynamicTemperatureController:
    """Adaptive temperature controller that recovers from loops and cools down to baseline."""

    def __init__(self, config: Optional[DynamicTemperatureConfig] = None):
        self.config = config or DynamicTemperatureConfig()
        self.state = DynamicTemperatureState(current_temperature=self.config.base_temperature)

    @property
    def current_temperature(self) -> float:
        return self.state.current_temperature

    def reset(self, base_temp: float = 0.7):
        self.config.base_temperature = base_temp
        self.state.current_temperature = base_temp
        self.state.recovery_attempts = 0
        self.state.tokens_since_adjustment = 0
        self.state.in_recovery = False
        self.state.last_reason = ""

    def update(self, verdict: GenerationLoopVerdict, token_id: int) -> float:
        if not self.config.enabled:
            return self.config.base_temperature

        if self.config.base_temperature <= 0.0 and not self.config.deterministic_mode:
            return 0.0

        self.state.tokens_since_adjustment += 1

        if (verdict.confidence in (LoopConfidence.SUSPICIOUS, LoopConfidence.PROBABLE_LOOP) or
            verdict.diversity_ratio < self.config.min_diversity):

            if ((not self.state.in_recovery or self.state.tokens_since_adjustment >= 4) and
                self.state.recovery_attempts < self.config.max_recovery_attempts and
                self.state.current_temperature < self.config.max_temperature):

                self.state.current_temperature = min(
                    self.config.max_temperature,
                    self.state.current_temperature + self.config.step
                )
                self.state.recovery_attempts += 1
                self.state.tokens_since_adjustment = 0
                self.state.in_recovery = True
                self.state.last_reason = f"Elevated temperature due to repetitive pattern ({verdict.reason})"
        elif verdict.confidence == LoopConfidence.NORMAL and self.state.in_recovery:
            if self.state.tokens_since_adjustment >= self.config.cooldown_tokens:
                if self.state.current_temperature > self.config.base_temperature:
                    self.state.current_temperature = max(
                        self.config.base_temperature,
                        self.state.current_temperature - self.config.decrease_rate
                    )
                if abs(self.state.current_temperature - self.config.base_temperature) < 1e-4:
                    self.state.current_temperature = self.config.base_temperature
                    self.state.in_recovery = False
                    self.state.recovery_attempts = 0

        return self.state.current_temperature
