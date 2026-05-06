from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class StatsTracker:
    """Records latency and estimated token counts for every LLM call."""

    def __post_init__(self):
        self._calls: list = []
        self._by_phase: dict = defaultdict(list)

    def record(self, phase: str, latency_ms: int, estimated_tokens: int) -> None:
        entry = {"phase": phase, "latency_ms": latency_ms, "estimated_tokens": estimated_tokens}
        self._calls.append(entry)
        self._by_phase[phase].append(entry)

    @property
    def total_calls(self) -> int:
        return len(self._calls)

    @property
    def total_latency_ms(self) -> int:
        return sum(c["latency_ms"] for c in self._calls)

    @property
    def total_tokens(self) -> int:
        return sum(c["estimated_tokens"] for c in self._calls)

    def summary(self) -> dict:
        by_phase = {}
        for phase, calls in self._by_phase.items():
            by_phase[phase] = {
                "calls": len(calls),
                "latency_ms": sum(c["latency_ms"] for c in calls),
                "tokens": sum(c["estimated_tokens"] for c in calls),
            }
        return {
            "total_calls": self.total_calls,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "by_phase": by_phase,
        }


# Module-level singleton — import this everywhere
tracker = StatsTracker()
