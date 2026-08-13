"""Pluggable sampling weights (design doc section 5: long-tail scheduling)."""

from __future__ import annotations

from typing import Protocol

from kgts.models import Node


class Prioritizer(Protocol):
    """Strategy interface: higher weight = more likely to be sampled."""

    def weight(self, node: Node) -> float: ...


class InverseFrequencyPrioritizer:
    """weight = 1 / (1 + alpha * times_sampled) -- counters high-frequency
    knowledge dominating the sample mix (K3 long-tail coverage)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def weight(self, node: Node) -> float:
        return 1.0 / (1.0 + self.alpha * node.stats.times_sampled)
