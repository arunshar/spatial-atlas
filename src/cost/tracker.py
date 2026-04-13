"""
Spatial Atlas — Token/Cost Budget Tracker

Tracks cumulative token usage and estimated cost across LLM calls.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("spatial-atlas.cost")


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_calls: int = 0
    estimated_cost_usd: float = 0.0


class CostTracker:
    def __init__(self, max_tokens: int = 150_000):
        self.max_tokens = max_tokens
        self.stats = UsageStats()

    def track(self, response) -> None:
        """Track usage from a litellm response."""
        usage = getattr(response, "usage", None)
        if not usage:
            return

        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = prompt + completion

        self.stats.prompt_tokens += prompt
        self.stats.completion_tokens += completion
        self.stats.total_tokens += total
        self.stats.num_calls += 1

        # Rough cost estimate
        cost = getattr(response, "_hidden_params", {}).get("response_cost", 0) or 0
        self.stats.estimated_cost_usd += cost

        logger.debug(
            f"Call #{self.stats.num_calls}: +{total} tokens "
            f"(total: {self.stats.total_tokens}/{self.max_tokens})"
        )

    def has_budget(self) -> bool:
        """Check if we're within token budget."""
        return self.stats.total_tokens < self.max_tokens

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.stats.total_tokens)

    def summary(self) -> str:
        return (
            f"Calls: {self.stats.num_calls}, "
            f"Tokens: {self.stats.total_tokens:,}/{self.max_tokens:,}, "
            f"Cost: ${self.stats.estimated_cost_usd:.4f}"
        )
