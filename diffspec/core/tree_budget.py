"""Adaptive draft-tree node budget controllers."""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from typing import Any


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _round_to_multiple(value: float, multiple: int) -> int:
    multiple = max(int(multiple), 1)
    return int(round(value / multiple) * multiple)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class HistoryTreeBudgetController:
    """Choose the next tree width from recent accept-length history.

    The default policy is a late-ramp schedule: the first window keeps the
    requested budget unchanged, then later decode steps receive a monotonically
    larger tree budget. Recent accept length still modulates the ramp so hard
    windows grow faster, but the per-request budget never shrinks.
    """

    base_nodes: int
    max_depth: int
    window_size: int = 10
    min_nodes: int | None = None
    max_nodes: int | None = None
    warmup_steps: int | None = None
    round_to: int = 4
    accept_history: list[int] = field(default_factory=list)
    budget_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.base_nodes = max(int(self.base_nodes), 1)
        self.max_depth = max(int(self.max_depth), 1)
        self.window_size = max(int(self.window_size), 1)
        self.warmup_steps = self.window_size if self.warmup_steps is None else max(int(self.warmup_steps), 0)
        self.min_nodes = self.base_nodes if self.min_nodes is None else max(int(self.min_nodes), 1)
        default_max = max(self.base_nodes, _round_to_multiple(self.base_nodes * 1.5, self.round_to))
        self.max_nodes = default_max if self.max_nodes is None else max(int(self.max_nodes), self.min_nodes)
        self.round_to = max(int(self.round_to), 1)
        self.current_nodes = self.base_nodes

    @classmethod
    def from_env(cls, base_nodes: int, max_depth: int) -> "HistoryTreeBudgetController | None":
        """Build a controller when requested through environment variables."""
        policy = os.environ.get("DIFFSPEC_TREE_BUDGET_POLICY", "history").strip().lower()
        if policy in {"", "fixed", "static", "none", "off", "0", "false"}:
            return None
        if policy not in {"history", "accept_history", "adaptive_history", "dynamic", "late_ramp"}:
            raise ValueError(f"Unknown DIFFSPEC_TREE_BUDGET_POLICY={policy!r}")

        window = _parse_int_env("DIFFSPEC_TREE_BUDGET_WINDOW", 10)
        min_nodes = _parse_int_env("DIFFSPEC_TREE_BUDGET_MIN", int(base_nodes))
        default_max = _round_to_multiple(int(base_nodes) * 1.5, _parse_int_env("DIFFSPEC_TREE_BUDGET_ROUND", 4))
        max_nodes = _parse_int_env("DIFFSPEC_TREE_BUDGET_MAX", int(default_max))
        warmup = _parse_int_env("DIFFSPEC_TREE_BUDGET_WARMUP", window)
        round_to = _parse_int_env("DIFFSPEC_TREE_BUDGET_ROUND", 4)
        return cls(
            base_nodes=base_nodes,
            max_depth=max_depth,
            window_size=window,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            warmup_steps=warmup,
            round_to=round_to,
        )

    @property
    def cache_nodes(self) -> int:
        """Largest tree width that may be requested by this controller."""
        return int(self.max_nodes)

    def observe(self, accept_length: int) -> int:
        """Record one accept length and return the next tree-node budget."""
        accept_length = max(int(accept_length), 0)
        self.accept_history.append(accept_length)
        step = len(self.accept_history)

        recent = self.accept_history[-self.window_size :]
        recent_avg = statistics.mean(recent) if recent else 0.0
        trend = 0.0
        if len(recent) >= 2:
            half = max(len(recent) // 2, 1)
            trend = statistics.mean(recent[half:]) - statistics.mean(recent[:half])

        if step <= self.warmup_steps or self.max_nodes <= self.min_nodes:
            budget = self.base_nodes
            pressure = 0.0
        else:
            late_pressure = _clamp((step - self.warmup_steps) / self.window_size, 0.0, 1.0)
            accept_gap = _clamp((self.max_depth - recent_avg) / self.max_depth, 0.0, 1.0)
            trend_pressure = _clamp(0.5 - trend / max(float(self.max_depth), 1.0), 0.0, 1.0)

            pressure = _clamp(
                0.70 * late_pressure + 0.20 * accept_gap + 0.10 * trend_pressure,
                0.0,
                1.0,
            )
            raw_budget = self.min_nodes + (self.max_nodes - self.min_nodes) * pressure
            budget = _round_to_multiple(raw_budget, self.round_to)
            budget = max(self.min_nodes, min(self.max_nodes, budget))
            budget = max(int(self.current_nodes), budget)

        self.current_nodes = int(budget)
        self.budget_history.append(
            {
                "step": step,
                "accept_length": accept_length,
                "recent_accept_avg": float(recent_avg),
                "recent_accept_trend": float(trend),
                "pressure": float(pressure),
                "budget": int(budget),
            }
        )
        return int(budget)

    def get_statistics(self) -> dict[str, Any]:
        budgets = [int(row["budget"]) for row in self.budget_history]
        accepts = [int(x) for x in self.accept_history]
        return {
            "policy": "late_ramp_history",
            "base_nodes": int(self.base_nodes),
            "min_nodes": int(self.min_nodes),
            "max_nodes": int(self.max_nodes),
            "window_size": int(self.window_size),
            "warmup_steps": int(self.warmup_steps),
            "current_nodes": int(self.current_nodes),
            "avg_budget": float(statistics.mean(budgets)) if budgets else float(self.base_nodes),
            "max_budget_seen": max(budgets) if budgets else int(self.base_nodes),
            "avg_accept_length": float(statistics.mean(accepts)) if accepts else 0.0,
            "history": list(self.budget_history),
        }

    def reset(self) -> None:
        self.accept_history.clear()
        self.budget_history.clear()
        self.current_nodes = self.base_nodes
