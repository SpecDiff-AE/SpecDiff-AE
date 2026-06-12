"""Runtime backends and target-model adapters for DiffSpec."""

from .attention_compat import FLASH_ATTN_AVAILABLE, PYTORCH_FLASH_ATTN_AVAILABLE
from .bundle_verifier import FusedBundleVerifier, PagedBundleVerifier
from .cache_state import initialize_past_key_values
from .draft_tree import DraftTree
from .tree_attention import (
    TRITON_AVAILABLE,
    prefix_attention_forward,
    tree_attention_forward,
)

__all__ = [
    "FLASH_ATTN_AVAILABLE",
    "PYTORCH_FLASH_ATTN_AVAILABLE",
    "FusedBundleVerifier",
    "PagedBundleVerifier",
    "initialize_past_key_values",
    "DraftTree",
    "TRITON_AVAILABLE",
    "prefix_attention_forward",
    "tree_attention_forward",
]
