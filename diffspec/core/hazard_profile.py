"""
Rejection-Hazard Profile Tracker

Implements the DiffSpec hazard-guided tree policy from Section 2.2.
The tracker keeps a sliding rejection-depth profile and allocates
branching budget toward depths where selected-path verification is more
likely to stop.
"""

import os

import torch
from typing import Dict, List
import numpy as np


class HazardProfileTracker:
    """Track selected-path stop depths and adapt per-depth branch budget.

    The tracker records the first rejected depth for recent speculative
    verification steps, smooths the resulting distribution with EMA, and
    concentrates branching budget near high-hazard depths while keeping
    other depths chain-like.
    
    Args:
        max_depth: Maximum speculation depth tracked by the profile.
        window_size: Number of recent rejection depths kept in the window.
        alpha: EMA smoothing coefficient in [0, 1].
        peak_budget_ratio: Reserved ratio for peak-depth allocation.
    """
    
    def __init__(
        self, 
        max_depth: int = 8, 
        window_size: int = 50,
        alpha: float = 0.2,
        peak_budget_ratio: float = 0.8,
        device: str = "cuda"
    ):
        self.max_depth = max_depth
        self.window_size = window_size
        self.alpha = alpha
        self.peak_budget_ratio = peak_budget_ratio
        self.device = device
        
        # Recent first-reject depths within the sliding window.
        self.reject_history = []
        
        # Per-depth rejection probability, initialized as a uniform prior.
        self.hazard_profile = torch.ones(max_depth, device=device) / max_depth
        
        # Running counters.
        self.total_rejections = 0
        self.depth_counts = torch.zeros(max_depth, device=device)
        
        # Cached branch budget. It is recomputed only after profile updates.
        self._cached_budget = None
        self._budget_dirty = True

        self.trace_enabled = os.environ.get("DIFFSPEC_HAZARD_TRACE", "").lower() in {"1", "true", "yes"}
        self.trace_records = []
    
    def update(self, first_reject_depth: int):
        """Record a new rejection depth and update the hazard profile.
        
        Args:
            first_reject_depth: Zero-based first rejected depth. Use -1 or
                max_depth when all speculative tokens are accepted.
        """
        raw_first_reject_depth = int(first_reject_depth)

        # Treat full acceptance as a late stop so it does not create a
        # synthetic early hazard.
        if first_reject_depth < 0:
            first_reject_depth = self.max_depth - 1
        
        # Clamp into the tracked depth range.
        first_reject_depth = min(first_reject_depth, self.max_depth - 1)
        
        # Update the sliding history.
        self.reject_history.append(first_reject_depth)
        if len(self.reject_history) > self.window_size:
            self.reject_history.pop(0)
        
        self.total_rejections += 1
        
        # Update lifetime depth counts.
        self.depth_counts[first_reject_depth] += 1
        
        # Compute the empirical distribution inside the active window.
        depth_counts_window = torch.zeros(self.max_depth, device=self.device)
        for d in self.reject_history:
            if 0 <= d < self.max_depth:
                depth_counts_window[d] += 1
        
        # Normalize into a probability distribution.
        total_count = len(self.reject_history)
        if total_count > 0:
            new_profile = depth_counts_window / total_count
        else:
            new_profile = torch.ones(self.max_depth, device=self.device) / self.max_depth
        
        # Smooth with an exponential moving average.
        self.hazard_profile = (
            self.alpha * new_profile + 
            (1 - self.alpha) * self.hazard_profile
        )

        if self.trace_enabled:
            p = self.hazard_profile.detach().float().cpu()
            entropy = float(-(p + 1e-10).mul(torch.log2(p + 1e-10)).sum().item())
            self.trace_records.append({
                "step": self.total_rejections,
                "raw_first_reject_depth": raw_first_reject_depth,
                "profile_depth": int(first_reject_depth),
                "full_accept": raw_first_reject_depth < 0,
                "hazard_profile": p.tolist(),
                "max_hazard_depth": int(torch.argmax(self.hazard_profile).item()),
                "max_hazard_prob": float(torch.max(self.hazard_profile).item()),
                "hazard_entropy": entropy,
            })
        
        # Mark the branch-budget cache dirty.
        self._budget_dirty = True
    
    def allocate_budget(
        self, 
        total_nodes: int,
        num_peaks: int = 2,
        min_branches: int = 1
    ) -> List[int]:
        """Allocate a per-depth branch budget from the hazard profile.

        High-hazard depths receive the extra budget. Other depths keep the
        minimum branching needed to preserve a connected chain.
        
        Args:
            total_nodes: Total node budget.
            num_peaks: Number of high-hazard depths to emphasize.
            min_branches: Minimum per-depth branching factor.
        
        Returns:
            Branch budget list of length ``max_depth``.
        """
        if not self._budget_dirty and self._cached_budget is not None:
            return self._cached_budget
        
        if len(self.reject_history) < 3:
            # Use a stable uniform allocation until enough feedback arrives.
            avg_branches = max(total_nodes // self.max_depth, min_branches)
            layer_budgets = [avg_branches] * self.max_depth
        else:
            # Baseline budget keeps all depths connected.
            layer_budgets = [min_branches] * self.max_depth
            
            peak_values, peak_depths = torch.topk(
                self.hazard_profile, 
                k=min(num_peaks, self.max_depth)
            )
            
            # Ignore very small peaks that are more likely to be noise.
            valid_peaks = peak_depths[peak_values > 0.05].cpu().tolist()
            
            if len(valid_peaks) == 0:
                avg_branches = max(total_nodes // self.max_depth, min_branches)
                layer_budgets = [avg_branches] * self.max_depth
            else:
                base_budget = min_branches * self.max_depth
                extra_budget = max(total_nodes - base_budget, 0)
                
                peak_weights = self.hazard_profile[valid_peaks]
                peak_weights = peak_weights / peak_weights.sum()
                
                for i, depth in enumerate(valid_peaks):
                    extra_nodes = int(extra_budget * peak_weights[i].item())
                    layer_budgets[depth] += extra_nodes
        
        # Keep the final allocation inside the requested node budget.
        total_allocated = sum(layer_budgets)
        if total_allocated > total_nodes:
            scale_factor = total_nodes / total_allocated
            layer_budgets = [max(int(b * scale_factor), min_branches) 
                           for b in layer_budgets]
        
        self._cached_budget = layer_budgets
        self._budget_dirty = False
        
        return layer_budgets

    def allocate_branch_caps(
        self,
        max_branching: int,
        num_peaks: int = 2,
        min_branching: int = 1,
        warmup_rejections: int = 5
    ) -> List[int]:
        """Return per-depth branch caps for utility-guided tree expansion.

        The tree builder consumes this as a per-parent top-k cap. During warmup
        or diffuse hazard profiles, keep the original full-width expansion so
        candidate quality does not regress before verifier feedback is useful.
        """
        max_branching = max(int(max_branching), 1)
        min_branching = max(1, min(int(min_branching), max_branching))

        if self.total_rejections < warmup_rejections or len(self.reject_history) < 3:
            return [max_branching] * self.max_depth

        if self.max_depth <= 0:
            return []

        depth_utility = torch.linspace(
            1.0,
            2.0,
            self.max_depth,
            device=self.hazard_profile.device,
            dtype=self.hazard_profile.dtype,
        )
        utility_profile = self.hazard_profile * depth_utility

        # If feedback is too flat, adaptive narrowing is more likely to hurt
        # acceptance than save useful compute.
        max_prob = float(torch.max(self.hazard_profile).item())
        if max_prob < (1.5 / self.max_depth):
            return [max_branching] * self.max_depth

        k = min(max(num_peaks, 1), self.max_depth)
        _, peak_depths = torch.topk(utility_profile, k=k)
        peak_set = {int(depth) for depth in peak_depths.cpu().tolist()}

        caps = [min_branching] * self.max_depth
        shoulder_cap = max(min_branching, max_branching // 2)
        for depth in peak_set:
            caps[depth] = max_branching
            if depth > 0:
                caps[depth - 1] = max(caps[depth - 1], shoulder_cap)
            if depth + 1 < self.max_depth:
                caps[depth + 1] = max(caps[depth + 1], shoulder_cap)

        return caps
    
    def get_dynamic_tree_config(
        self, 
        total_nodes: int = 50,
        num_peaks: int = 2
    ) -> Dict:
        """Return the current dynamic tree configuration.
        
        Returns:
            Dictionary containing branch budgets, the smoothed hazard profile,
            peak depths, and rejection-history metadata.
        """
        layer_budgets = self.allocate_budget(total_nodes, num_peaks)
        
        peak_values, peak_depths = torch.topk(
            self.hazard_profile, 
            k=min(num_peaks, self.max_depth)
        )
        
        return {
            'layer_budgets': layer_budgets,
            'hazard_profile': self.hazard_profile.cpu().tolist(),
            'peak_depths': peak_depths.cpu().tolist(),
            'peak_values': peak_values.cpu().tolist(),
            'total_nodes': sum(layer_budgets),
            'total_rejections': self.total_rejections,
            'history_size': len(self.reject_history)
        }
    
    def reset(self):
        """Reset the tracker for a new sequence or conversation."""
        self.reject_history = []
        self.hazard_profile = torch.ones(
            self.max_depth, device=self.device
        ) / self.max_depth
        self.depth_counts.zero_()
        self.total_rejections = 0
        self.trace_records = []
        self._budget_dirty = True
        self._cached_budget = None
    
    def get_statistics(self) -> Dict:
        """Return aggregate profile statistics."""
        if len(self.reject_history) == 0:
            avg_depth = 0.0
            std_depth = 0.0
        else:
            avg_depth = np.mean(self.reject_history)
            std_depth = np.std(self.reject_history)
        
        stats = {
            'total_rejections': self.total_rejections,
            'window_size': len(self.reject_history),
            'avg_reject_depth': avg_depth,
            'std_reject_depth': std_depth,
            'hazard_entropy': self._compute_entropy(),
            'max_hazard_depth': int(torch.argmax(self.hazard_profile).item()),
            'max_hazard_prob': float(torch.max(self.hazard_profile).item())
        }
        if self.trace_enabled:
            stats['trace_records'] = self.trace_records.copy()
            stats['final_reject_history'] = self.reject_history.copy()
            stats['final_hazard_profile'] = self.hazard_profile.detach().float().cpu().tolist()
            stats['depth_counts'] = self.depth_counts.detach().float().cpu().tolist()
        return stats
    
    def _compute_entropy(self) -> float:
        """Compute the entropy of the current hazard profile."""
        p = self.hazard_profile + 1e-10
        entropy = -torch.sum(p * torch.log2(p)).item()
        return entropy
    
    def visualize_profile(self) -> str:
        """Render the hazard profile as an ASCII bar chart."""
        lines = ["Hazard Profile:"]
        max_bar_width = 50
        
        max_prob = float(torch.max(self.hazard_profile).item())
        
        for depth in range(self.max_depth):
            prob = float(self.hazard_profile[depth].item())
            bar_width = int((prob / (max_prob + 1e-10)) * max_bar_width)
            bar = "#" * bar_width
            lines.append(f"  Depth {depth}: {bar} {prob:.3f}")
        
        return "\n".join(lines)
    
    def save_state(self) -> Dict:
        """Serialize the tracker state for checkpointing."""
        return {
            'reject_history': self.reject_history.copy(),
            'hazard_profile': self.hazard_profile.cpu().numpy(),
            'depth_counts': self.depth_counts.cpu().numpy(),
            'total_rejections': self.total_rejections
        }
    
    def load_state(self, state: Dict):
        """Load a tracker state produced by :meth:`save_state`."""
        self.reject_history = state['reject_history']
        self.hazard_profile = torch.tensor(
            state['hazard_profile'], device=self.device
        )
        self.depth_counts = torch.tensor(
            state['depth_counts'], device=self.device
        )
        self.total_rejections = state['total_rejections']
        self._budget_dirty = True
