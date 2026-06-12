"""Optimized draft-tree construction for DiffSpec.

The public Tree interface is intentionally compact: ``initialize``, ``add``,
``generate_attention_mask``, and ``update``. Internally the builder maintains
rolling top-k state, reuses tensor buffers, and constructs attention masks
with vectorized row slices to reduce allocation and indexing overhead.
"""

import os

import torch


def _branch_policy_caps(policy, nnodes, max_depth):
    policy = (policy or "online").strip().lower().replace("-", "_")
    if policy in {"online", "adaptive", "hazard", "hazard_online"}:
        return "online", None
    if policy in {"full", "static_full", "none"}:
        return "full", None
    if policy in {"chain", "static_chain"}:
        return "chain", [1] * max_depth
    if policy.startswith("uniform"):
        suffix = policy.replace("uniform", "").replace("_", "").replace(":", "")
        cap = int(suffix) if suffix else max(1, min(nnodes, 4))
        return f"uniform{cap}", [max(1, min(nnodes, cap))] * max_depth
    if policy in {"shallow", "front", "front_loaded"}:
        caps = [1] * max_depth
        if max_depth > 0:
            caps[0] = nnodes
        if max_depth > 1:
            caps[1] = max(1, nnodes // 2)
        return "shallow", caps
    if policy in {"deep", "back", "back_loaded"}:
        caps = [1] * max_depth
        if max_depth > 1:
            caps[-2] = max(1, nnodes // 2)
        if max_depth > 0:
            caps[-1] = nnodes
        return "deep", caps
    raise ValueError(f"Unknown DIFFSPEC_BRANCH_POLICY={policy!r}")


class DraftTree:
    def __init__(self, nnodes, device, threshold, max_depth, hazard_tracker=None):
        self.nnodes = int(nnodes)
        self.device = device
        self.threshold = float(threshold)
        self.max_depth = int(max_depth)
        
        # Hazard-driven tree optimization. DIFFSPEC_BRANCH_POLICY is an
        # experiment hook: static policies still allow hazard tracing, but do
        # not let the online profile affect branch caps.
        self.hazard_tracker = hazard_tracker
        self.branch_policy, static_caps = _branch_policy_caps(
            os.environ.get("DIFFSPEC_BRANCH_POLICY", "online"),
            self.nnodes,
            self.max_depth,
        )
        self.use_dynamic_budget = hazard_tracker is not None and self.branch_policy == "online"
        if static_caps is not None:
            self._branch_caps = static_caps
        elif self.use_dynamic_budget and hasattr(self.hazard_tracker, "allocate_branch_caps"):
            self._branch_caps = self.hazard_tracker.allocate_branch_caps(
                max_branching=self.nnodes,
                num_peaks=2,
                min_branching=1,
            )
        else:
            self._branch_caps = None

        # Build progress.
        self.depth = 0
        self.weight = 0.0

        # Public matrices retained for compatibility with callers that inspect
        # the tree state directly.
        self.weight_matrix    = torch.zeros([self.max_depth, self.nnodes], device=self.device)             # float32
        self.input_ids_matrix = torch.full ([self.max_depth, self.nnodes], -1,  dtype=torch.long, device=self.device)
        # parents_matrix[layer, col] = parent column from the previous layer.
        self.parents_matrix   = torch.full ([self.max_depth, self.nnodes], -1,  dtype=torch.long, device=self.device)

        # Rolling state used to build draft attention masks.
        self.kv_mask       = torch.zeros([self.nnodes, 0], dtype=torch.int8, device=self.device)           # accumulated ancestors
        self.tri           = torch.eye(self.nnodes, dtype=torch.int8, device=self.device)                  # same-layer self attention
        self.rows          = torch.arange(self.nnodes, device=self.device)
        self.position_id   = torch.zeros(self.nnodes, dtype=torch.long, device=self.device)
        self.kv_cache_mask = torch.zeros([self.nnodes, self.nnodes], dtype=torch.int8, device=self.device) # parent one-hot

        # Running global top-k values and linear indices.
        self._run_top_vals = torch.empty((0,), device=self.device)
        self._run_top_idx  = torch.empty((0,), dtype=torch.long, device=self.device)

        # Reusable buffers for final masks and intermediate construction.
        max_cols = self.max_depth * self.nnodes
        self._final_attn     = torch.zeros((self.nnodes, self.nnodes), dtype=torch.int8, device=self.device)
        self._parent_pos_buf = torch.full((self.nnodes,), -1, dtype=torch.long, device=self.device)
        self._diag           = torch.arange(self.nnodes, device=self.device)
        self._new_kv_mask    = torch.zeros((self.nnodes, max_cols), dtype=torch.int8, device=self.device)
        # Step attention = new KV visibility plus same-layer self attention.
        self._attn_step      = torch.zeros((self.nnodes, max_cols + self.nnodes), dtype=torch.int8, device=self.device)

    @torch.no_grad()
    def initialize(self, logits):
        """Initialize depth 0 from the last-token distribution."""
        last = logits[0, -1] if logits.dim() == 3 else logits[0][-1]
        topv, ids = torch.topk(last, k=self.nnodes, dim=-1)  # [nnodes]

        self.weight_matrix[0].copy_(topv)
        self.input_ids_matrix[0].copy_(ids)
        # Depth 0 has no semantic parent; keep row ids for shape compatibility.
        self.parents_matrix[0].copy_(self.rows)

        self._run_top_vals = topv.clone()
        self._run_top_idx  = 0 * self.nnodes + torch.arange(self.nnodes, device=self.device)

        out = {
            "input_ids":      ids,                          # [nnodes]
            "position_ids":   self.position_id + 1,         # [nnodes]
            "attention_mask": self.tri,                     # [nnodes, nnodes]
            "parent_last":    self.rows,                    # [nnodes]
            "is_final":       False
        }
        self.depth = 1
        return out

    @torch.no_grad()
    def add(self, logits):
        """Expand one depth by per-parent top-k, then keep global top-k."""
        cur = logits[0] if logits.dim() == 3 else logits[0]      # [nnodes, vocab]
        
        # Select this depth's branch cap.
        if self._branch_caps is not None:
            k_per_node = self._branch_caps[self.depth] if self.depth < len(self._branch_caps) else self.nnodes
        else:
            k_per_node = self.nnodes
        
        topv, ids = torch.topk(cur, k=k_per_node, dim=-1)       # [nnodes, k_per_node]

        last_w = self.weight_matrix[self.depth - 1].unsqueeze(1) # [nnodes,1]
        topv = topv * last_w

        flat_v  = topv.reshape(-1)
        flat_id = ids.reshape(-1)
        gvals, gidx = torch.topk(flat_v, k=self.nnodes, dim=-1)  # [nnodes]
        input_ids = flat_id[gidx]                                 # current-depth token
        parents   = (gidx // k_per_node).to(torch.long)           # parent column from previous depth

        self.weight_matrix[self.depth].copy_(gvals)
        self.input_ids_matrix[self.depth].copy_(input_ids)
        self.parents_matrix[self.depth].copy_(parents)

        # Build the draft attention mask by inheriting parent ancestors, adding
        # the immediate parent column, and appending same-depth self attention.
        prev_cols = self.kv_mask.size(1)
        new_cols = prev_cols + self.nnodes
        new_kv_mask = self._new_kv_mask[:, :new_cols]
        if prev_cols > 0:
            new_kv_mask[:, :prev_cols] = self.kv_mask.index_select(0, parents)

        self.kv_cache_mask.zero_()
        self.kv_cache_mask[self.rows, parents] = 1
        new_kv_mask[:, prev_cols:new_cols] = self.kv_cache_mask

        attn_step = self._attn_step[:, :new_cols + self.nnodes]
        attn_step[:, :new_cols] = new_kv_mask
        attn_step[:, new_cols:] = self.tri

        self.kv_mask = new_kv_mask

        cur_lin_idx = self.depth * self.nnodes + torch.arange(self.nnodes, device=self.device)
        cand_vals = torch.cat([self._run_top_vals, self.weight_matrix[self.depth]])
        cand_idx  = torch.cat([self._run_top_idx,  cur_lin_idx])
        keep_vals, keep_pos = torch.topk(cand_vals, k=self.nnodes)
        self._run_top_vals = keep_vals
        self._run_top_idx  = cand_idx[keep_pos]

        out = {
            "input_ids":      input_ids,                          # [nnodes]
            "position_ids":   self.position_id + (self.depth + 1),# [nnodes]
            "attention_mask": attn_step,                          # [nnodes, new_cols+nnodes]
            "parent_last":    parents,                             # [nnodes]
            "is_final":       False
        }
        self.depth += 1
        return out

    @torch.no_grad()
    def generate_attention_mask(self, parents_index: torch.Tensor):
        """Generate the final ancestor-visible attention mask.

        ``parents_index`` stores each final node's parent position in the
        final sequence, or -1 when the parent is absent.
        """
        K = int(parents_index.numel())
        attn = self._final_attn
        attn.zero_()

        diag = self._diag[:K]
        attn[diag, diag] = 1

        cur = parents_index
        max_hops = max(self.depth - 1, 0)
        for _ in range(max_hops):
            valid = (cur >= 0) & (cur < K)
            if not bool(valid.any()):
                break
            rows = diag[valid]
            cols = cur[valid]
            attn[rows, cols] = 1
            nxt = self._parent_pos_buf
            nxt.fill_(-1)
            nxt[valid] = parents_index[cols]
            cur = nxt.clone()
        return attn[:K, :K]

    @torch.no_grad()
    def update(self, logits):
        if self.depth == 0:
            return self.initialize(logits)

        step_out = self.add(logits)

        # Continue while the running global top-k mass improves enough.
        new_total = float(self._run_top_vals.sum())
        if (new_total - self.weight > self.threshold) and (self.depth < self.max_depth):
            self.weight = new_total
            return step_out

        top_index = self._run_top_idx  # [K]
        rows = (top_index // self.nnodes).to(torch.long)
        cols = (top_index %  self.nnodes).to(torch.long)

        # Preserve layer order for compatibility with downstream decoding.
        order = torch.argsort(rows)
        rows = rows[order]
        cols = cols[order]

        K = self.nnodes
        pos_table = torch.full((self.depth, self.nnodes), -1, dtype=torch.long, device=self.device)
        pos_table[rows, cols] = torch.arange(K, device=self.device)

        parent_pos = torch.full((K,), -1, dtype=torch.long, device=self.device)
        par_cols = self.parents_matrix[rows, cols]
        has_parent = (rows > 0) & (par_cols >= 0)
        if bool(has_parent.any()):
            pr = rows[has_parent] - 1
            pc = par_cols[has_parent]
            mapped = pos_table[pr, pc]
            parent_pos[has_parent] = mapped

        attn_mask = self.generate_attention_mask(parent_pos)

        outputs = {
            "input_ids":      self.input_ids_matrix[rows, cols],    # [K]
            "position_ids":   rows + 1,                             # [K]
            "attention_mask": attn_mask,                            # [K, K]
            "parent_last":    parent_pos,                           # [K]
            "is_final":       True
        }
        return outputs
