"""
Ancestor-Based Bundle Scheduler

Implements the batch-scheduled verification grouping described in
DiffSpec Section 3.2. Leaves are grouped by their nearest branching
ancestor, then split into fixed-size bundles so verifier kernels can
reuse shared prefixes.
"""

import torch
from typing import Dict, List
from collections import defaultdict


class BundleScheduler:
    """Group tree leaves into prefix-coherent verification bundles.

    Leaves that share the same nearest branching ancestor are placed in the
    same bucket. Each bucket is then chunked into fixed-size bundles for
    batch verification.
    
    Args:
        bundle_size: Fixed bundle size.
        max_bundles: Maximum number of bundles to emit.
    """
    
    def __init__(
        self, 
        bundle_size: int = 3,
        max_bundles: int = 100,
        device: str = "cuda"
    ):
        self.bundle_size = bundle_size
        self.max_bundles = max_bundles
        self.device = device
        
        # Running scheduling statistics.
        self.total_scheduled = 0
        self.total_bundles = 0
        self.avg_prefix_length = 0.0
    
    def schedule_bundles(
        self, 
        tree_descriptor: Dict,
        return_tensors: bool = True
    ) -> List[Dict]:
        """Schedule tree leaves into prefix-coherent bundles.
        
        Args:
            tree_descriptor: Dictionary with ``leaf_ids``, ``parent_map``,
                optional ``depth_map``, and optional ``token_map`` entries.
            return_tensors: Return tensor-valued bundles when true.
        
        Returns:
            A list of bundle dictionaries containing leaves, branching
            ancestor, shared prefix, prefix length, and bundle id.
        """
        leaf_ids = tree_descriptor['leaf_ids']
        parent_map = tree_descriptor['parent_map']
        depth_map = tree_descriptor.get('depth_map', {})
        
        # Normalize tensor inputs for Python-side grouping.
        if isinstance(leaf_ids, torch.Tensor):
            leaf_ids = leaf_ids.cpu().tolist()
        
        leaf_to_branc = {}
        for leaf in leaf_ids:
            br_anc = self._find_branching_ancestor(leaf, parent_map)
            leaf_to_branc[leaf] = br_anc
        
        buckets = defaultdict(list)
        for leaf, br_anc in leaf_to_branc.items():
            buckets[br_anc].append(leaf)
        
        bundles = []
        bundle_id = 0
        
        for br_anc, bucket_leaves in buckets.items():
            # Sort by depth when available to improve locality.
            if depth_map:
                bucket_leaves = sorted(
                    bucket_leaves, 
                    key=lambda x: depth_map.get(x, 0)
                )
            
            for i in range(0, len(bucket_leaves), self.bundle_size):
                bundle_leaves = bucket_leaves[i:i+self.bundle_size]
                
                if bundle_leaves:
                    shared_prefix = self._get_path_to_ancestor(
                        bundle_leaves[0], br_anc, parent_map
                    )
                else:
                    shared_prefix = []
                
                bundle = {
                    'leaves': bundle_leaves,
                    'br_anc': br_anc,
                    'shared_prefix': shared_prefix,
                    'prefix_length': len(shared_prefix),
                    'bundle_id': bundle_id,
                    'bucket_id': br_anc
                }
                
                bundles.append(bundle)
                bundle_id += 1
                
                if len(bundles) >= self.max_bundles:
                    break
            
            if len(bundles) >= self.max_bundles:
                break
        
        # Update aggregate statistics.
        self.total_scheduled += len(leaf_ids)
        self.total_bundles += len(bundles)
        if bundles:
            self.avg_prefix_length = sum(b['prefix_length'] for b in bundles) / len(bundles)
        
        if return_tensors:
            bundles = self._convert_to_tensors(bundles)
        
        return bundles
    
    def _find_branching_ancestor(
        self, 
        node: int, 
        parent_map: Dict[int, int]
    ) -> int:
        """Walk upward and return the nearest ancestor with multiple children."""
        children_count = defaultdict(int)
        for child, parent in parent_map.items():
            if parent >= 0:
                children_count[parent] += 1
        
        current = node
        while current in parent_map:
            parent = parent_map[current]
            if parent < 0:
                return current
            if children_count[parent] > 1:
                return parent
            current = parent
        
        return 0
    
    def _get_path_to_ancestor(
        self, 
        node: int, 
        ancestor: int, 
        parent_map: Dict[int, int]
    ) -> List[int]:
        """Return the path from ``ancestor`` to ``node``.
        
        Returns:
            Path ordered as ``[ancestor, ..., node]``.
        """
        path = []
        current = node
        
        while current != ancestor and current in parent_map:
            path.append(current)
            parent = parent_map[current]
            if parent < 0:
                break
            current = parent
        
        path.append(ancestor)
        path.reverse()
        
        return path
    
    def _convert_to_tensors(self, bundles: List[Dict]) -> List[Dict]:
        """Convert list-valued bundle fields to tensors."""
        tensor_bundles = []
        
        for bundle in bundles:
            tensor_bundle = {
                'leaves': torch.tensor(
                    bundle['leaves'], 
                    dtype=torch.long, 
                    device=self.device
                ),
                'br_anc': bundle['br_anc'],
                'shared_prefix': torch.tensor(
                    bundle['shared_prefix'], 
                    dtype=torch.long, 
                    device=self.device
                ) if bundle['shared_prefix'] else torch.tensor([], dtype=torch.long, device=self.device),
                'prefix_length': bundle['prefix_length'],
                'bundle_id': bundle['bundle_id'],
                'bucket_id': bundle['bucket_id']
            }
            tensor_bundles.append(tensor_bundle)
        
        return tensor_bundles
    
    def create_bundle_attention_masks(
        self, 
        bundles: List[Dict],
        max_seq_len: int
    ) -> List[torch.Tensor]:
        """Create one attention mask per bundle.
        
        Args:
            bundles: Output produced by :meth:`schedule_bundles`.
            max_seq_len: Maximum sequence length for each mask row.
        
        Returns:
            List of ``[bundle_size, max_seq_len]`` masks.
        """
        masks = []
        
        for bundle in bundles:
            leaves = bundle['leaves']
            prefix = bundle['shared_prefix']
            
            if isinstance(leaves, torch.Tensor):
                bundle_size = leaves.size(0)
            else:
                bundle_size = len(leaves)
            
            mask = torch.zeros(
                bundle_size, max_seq_len, 
                dtype=torch.int8, 
                device=self.device
            )
            
            # The shared prefix is visible to every bundle member.
            if isinstance(prefix, torch.Tensor) and prefix.numel() > 0:
                prefix_indices = prefix
                mask[:, prefix_indices] = 1
            elif isinstance(prefix, list) and len(prefix) > 0:
                prefix_indices = torch.tensor(prefix, device=self.device)
                mask[:, prefix_indices] = 1
            
            # Full ancestral-path masks can be added here once the verifier
            # consumes per-leaf path metadata.
            
            masks.append(mask)
        
        return masks
    
    def merge_bundle_logits(
        self, 
        bundle_logits: List[torch.Tensor],
        bundles: List[Dict]
    ) -> torch.Tensor:
        """Merge logits produced for individual bundles.
        
        Args:
            bundle_logits: List of ``[bundle_size, vocab_size]`` tensors.
            bundles: Bundle descriptors corresponding to each logits tensor.
        
        Returns:
            A ``[total_leaves, vocab_size]`` tensor.
        """
        if not bundle_logits:
            return torch.empty(0, device=self.device)
        
        sorted_pairs = sorted(
            zip(bundles, bundle_logits), 
            key=lambda x: x[0]['bundle_id']
        )
        
        all_logits = [logits for _, logits in sorted_pairs]
        merged = torch.cat(all_logits, dim=0)
        
        return merged
    
    def get_statistics(self) -> Dict:
        """Return aggregate scheduling statistics."""
        return {
            'total_scheduled_leaves': self.total_scheduled,
            'total_bundles_created': self.total_bundles,
            'avg_bundle_size': (
                self.total_scheduled / self.total_bundles 
                if self.total_bundles > 0 else 0
            ),
            'avg_prefix_length': self.avg_prefix_length,
            'bundle_size_config': self.bundle_size
        }
    
    def reset_statistics(self):
        """Reset aggregate scheduling statistics."""
        self.total_scheduled = 0
        self.total_bundles = 0
        self.avg_prefix_length = 0.0


class TreeDescriptorBuilder:
    """Build scheduler descriptors from tree representations."""
    
    @staticmethod
    def from_draft_tree(tree, final_indices: torch.Tensor = None) -> Dict:
        """Extract a scheduler descriptor from a ``DraftTree`` instance.
        
        Args:
            tree: Tree instance with ``depth``, ``nnodes``, and parent matrix.
            final_indices: Optional selected node indices.
        
        Returns:
            Descriptor accepted by :meth:`BundleScheduler.schedule_bundles`.
        """
        if final_indices is None:
            depth = tree.depth
            nnodes = tree.nnodes
            all_indices = []
            for d in range(depth):
                for n in range(nnodes):
                    all_indices.append(d * nnodes + n)
            final_indices = torch.tensor(all_indices, device=tree.device)
        
        parent_map = {}
        depth_map = {}
        
        for idx in final_indices.cpu().tolist():
            row = idx // tree.nnodes
            col = idx % tree.nnodes
            
            node_id = idx
            depth_map[node_id] = row
            
            if row > 0:
                parent_col = tree.parents_matrix[row, col].item()
                if parent_col >= 0:
                    parent_id = (row - 1) * tree.nnodes + parent_col
                    parent_map[node_id] = parent_id
                else:
                    parent_map[node_id] = -1
            else:
                parent_map[node_id] = -1
        
        all_nodes = set(final_indices.cpu().tolist())
        parent_nodes = set(parent_map.values()) - {-1}
        leaf_nodes = list(all_nodes - parent_nodes)
        
        return {
            'leaf_ids': leaf_nodes,
            'parent_map': parent_map,
            'depth_map': depth_map,
            'all_nodes': list(all_nodes)
        }
    
    @staticmethod
    def from_parent_tensor(
        parent: torch.Tensor,
        num_nodes: int = None
    ) -> Dict:
        """Build a scheduler descriptor from a parent-index tensor.
        
        Args:
            parent: ``[N]`` or ``[N-1]`` parent index tensor.
            num_nodes: Optional total number of nodes.
        
        Returns:
            Descriptor accepted by :meth:`BundleScheduler.schedule_bundles`.
        """
        if num_nodes is None:
            num_nodes = parent.size(0)
        
        parent_map = {}
        for i in range(num_nodes):
            if i < parent.size(0):
                parent_map[i] = int(parent[i].item())
            else:
                parent_map[i] = -1
        
        all_nodes = set(range(num_nodes))
        parent_nodes = set(p for p in parent_map.values() if p >= 0)
        leaf_nodes = list(all_nodes - parent_nodes)
        
        # Compute depth by walking the parent chain.
        depth_map = {}
        for node in range(num_nodes):
            depth = 0
            current = node
            visited = set()
            while current in parent_map and parent_map[current] >= 0:
                if current in visited:
                    break
                visited.add(current)
                current = parent_map[current]
                depth += 1
            depth_map[node] = depth
        
        return {
            'leaf_ids': leaf_nodes,
            'parent_map': parent_map,
            'depth_map': depth_map,
            'all_nodes': list(all_nodes)
        }
