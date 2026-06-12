import copy
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from termcolor import colored
from transformers import AutoConfig, AutoTokenizer

from ..core.chunk_encoding import SalienceAwareChunkEncoder
from ..core.hazard_profile import HazardProfileTracker
from ..core.tree_budget import HistoryTreeBudgetController
from ..core.bundle_scheduler import BundleScheduler
from ..core.kv_cache import PagedKVCache, PagedKVConfig
from ..core.kv_cache.chunk_arena import ChunkArena, ChunkArenaConfig
from ..amd import (
    AmdArenaConfig,
    AmdChunkArena,
    AmdResidencyConfig,
    is_rocm_pytorch,
)
from ..runtime.bundle_verifier import FusedBundleVerifier
from ..runtime.attention_compat import (
    FLASH_ATTN_AVAILABLE,
    PYTORCH_FLASH_ATTN_AVAILABLE,
    sdpa_backend_name,
)
from ..runtime.tree_attention import TRITON_AVAILABLE
from ..runtime.cache_state import initialize_past_key_values
from ..runtime.llama_target_model import LlamaForCausalLM as KVLlamaForCausalLM
from ..runtime.qwen3_target_model import Qwen3ForCausalLM as KVQwen3ForCausalLM
from ..runtime.qwen2_target_model import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .draft_network import DraftNetwork
from .draft_config import DraftConfig
from .tree_verification import (
    prepare_logits_processor,
    print_newly_accepted_tokens,
    tree_decoding,
    verify,
)

def tensor_size_bytes(tensor):
    return tensor.element_size() * tensor.nelement()

def total_kv_cache_size(past_key_values):
    total = 0
    for layer in past_key_values:
        for kv in layer:
            # Adjust "kv.data" if your KVCache stores its tensor differently.
            total += tensor_size_bytes(kv.data)
    return total

def _ncu_phase_requested(phase):
    raw = os.environ.get("DIFFSPEC_NCU_PROFILE_PHASE", "")
    requested = {part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()}
    phase = phase.lower()
    return (
        "all" in requested
        or phase in requested
        or (phase.endswith("_decode") and "decode" in requested)
    )

def _maybe_cuda_profiler_start(phase):
    if not _ncu_phase_requested(phase) or not torch.cuda.is_available():
        return False
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    return True

def _maybe_cuda_profiler_stop(active):
    if active and torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStop()


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}

class DiffSpecModel(nn.Module):
    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            draft_config_path,
            draft_state_dict
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = DraftConfig.from_pretrained(draft_config_path)
        with open(draft_config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        bias = raw_config.get("bias", True)

        # Keep draft position encoding aligned with the target model.
        target_cfg = getattr(base_model, "config", None)
        if target_cfg is not None:
            if getattr(target_cfg, "rope_scaling", None) is not None:
                config.rope_scaling = copy.deepcopy(target_cfg.rope_scaling)
            if hasattr(target_cfg, "max_position_embeddings"):
                config.max_position_embeddings = target_cfg.max_position_embeddings
            if hasattr(target_cfg, "rope_theta"):
                config.rope_theta = target_cfg.rope_theta
        self.draft_model = DraftNetwork(config, bias=bias, load_emb=True, path=base_model_name_or_path)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.draft_model.diff_device = True
            if not low_memory:
                # self.draft_model.head=nn.Linear(base_model.lm_head.in_features,base_model.lm_head.out_features,bias=False)
                # self.draft_model.head.weight=copy.deepcopy(base_model.lm_head.weight)
                # self.draft_model.head.to(device)
                self.draft_model.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.draft_model.layer_device = device

        else:
            self.draft_model.diff_device = False
        if config.vocab_size == config.draft_vocab_size:
            del self.draft_model.d2t, self.draft_model.t2d
        self.draft_model.load_state_dict(draft_state_dict, strict=False)
        self.draft_model.to(self.base_model.dtype).to(device)
        self.draft_model.tokenizer = self.tokenizer

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            base_model_path=None,
            draft_model_path=None,
            **kwargs,
    ):
        architecture = AutoConfig.from_pretrained(base_model_path).architectures[0]
        if architecture == "LlamaForCausalLM":
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif architecture == "Qwen3ForCausalLM":
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif architecture == "Qwen2ForCausalLM":
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            raise ValueError(f"Unsupported target architecture: {architecture}")
    
        download_kwargs = {}
        hf_cache_dir = os.environ.get("DIFFSPEC_HF_CACHE")
        if hf_cache_dir:
            download_kwargs["cache_dir"] = hf_cache_dir

        config_path = os.path.join(draft_model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = hf_hub_download(draft_model_path, "config.json", **download_kwargs)
        draft_weights_path = os.path.join(draft_model_path, "pytorch_model.bin")
        if not os.path.exists(draft_weights_path):
            try:
                draft_weights_path = hf_hub_download(draft_model_path, "pytorch_model.bin", **download_kwargs)
            except Exception:
                draft_weights_path = hf_hub_download(draft_model_path, "model.safetensors", **download_kwargs)
        if draft_weights_path.endswith(".safetensors"):
            draft_state_dict = load_safetensors(draft_weights_path, device=str(base_model.device))
        else:
            draft_state_dict = torch.load(draft_weights_path, map_location=base_model.device)
        model = cls(
            base_model,
            base_model_path,
            config_path,
            draft_state_dict
        )

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            tree_attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            nodes=None,
            threshold=None,
            max_depth=None,
            logits_processor=None,
            retrieve_attn_scores=False,
            best=True
    ):

        with torch.inference_mode():
            # Prefill target model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tree_attention_mask=tree_attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_attentions=True,
                init=init,
                target_use_flash_prefill = self.draft_model.target_use_flash_prefill,
                target_use_hybrid_tree_attn = self.draft_model.target_use_hybrid_tree_attn,
                retrieve_attn_scores=retrieve_attn_scores,
                best=best
            )
            
            if output_orig:
                orig_input = outputs[0][:, -1:] if init else outputs[0]
                orig = self.base_model.lm_head(orig_input)
            hidden_states = outputs[0] if init else outputs[0].clone()
            
            if self.draft_model.use_retrieval_cache:
                if retrieve_attn_scores:
                    # Keep only the final target attention layer for retrieval scoring.
                    self.draft_model.attn_scores = outputs.attentions[-1]
             

        
        
        # initial tree draft
        if init:
            
            
            
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # Clone the output hidden states
            draft_device = self.draft_model.lm_head.weight.device
            if outputs["hidden_states"][0].device != draft_device:
                outputs["hidden_states"] = [x.to(draft_device) for x in outputs["hidden_states"]]
            hidden_states=torch.cat(outputs["hidden_states"],dim=-1)
            input_ids,position_ids,tree_attention_mask,parent = self.draft_model.build_speculative_tree(hidden_states, input_ids, self.base_model.lm_head, nodes=nodes,threshold=threshold,max_depth=max_depth)
            return input_ids,position_ids,tree_attention_mask,token,parent
        else:

            return outputs, orig, hidden_states

    @torch.no_grad()
    def diffspec_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=256,
            nodes=48,
            threshold=0.12,
            max_depth=6,
            output_result_line=False,
            verbose=True,
            retrieval_verbose=False,
            use_diffspec=None,
            diffspec_plugins=None,
            retrieval_chunk_size = 32,
            retrieve_top_k = 32,
            retrieve_every_n_steps = 4,
            retrieval_min_context = 50000,
            return_generated_text=False,
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_len = input_ids.shape[1]

        if use_diffspec is None:
            use_diffspec = [1, 1, 1, 1, 1]
        if len(use_diffspec) != 5:
            raise ValueError("use_diffspec must contain 5 flags")

        requested_retrieval_cache = bool(use_diffspec[0])
        self.draft_model.use_retrieval_cache = requested_retrieval_cache
        self.draft_model.target_use_flash_prefill = bool(use_diffspec[1])
        requested_hybrid_tree_attn = bool(use_diffspec[2])
        hybrid_disabled_reason = None
        self.draft_model.target_use_hybrid_tree_attn = requested_hybrid_tree_attn
        hybrid_tree_available = bool(
            FLASH_ATTN_AVAILABLE or (PYTORCH_FLASH_ATTN_AVAILABLE and TRITON_AVAILABLE)
        )
        if requested_hybrid_tree_attn and not hybrid_tree_available:
            self.draft_model.target_use_hybrid_tree_attn = False
            if not PYTORCH_FLASH_ATTN_AVAILABLE and not FLASH_ATTN_AVAILABLE:
                hybrid_disabled_reason = "flash_attention_unavailable"
            elif not TRITON_AVAILABLE:
                hybrid_disabled_reason = "triton_unavailable"
            else:
                hybrid_disabled_reason = "hybrid_backend_unavailable"
        self.draft_model.draft_use_flash_prefill = bool(use_diffspec[3])
        self.draft_model.best = bool(use_diffspec[4])

        retrieval_disabled_reason = None
        if self.draft_model.use_retrieval_cache and retrieval_min_context:
            if input_len < int(retrieval_min_context):
                self.draft_model.use_retrieval_cache = False
                retrieval_disabled_reason = (
                    f"input_len<{int(retrieval_min_context)}"
                )

        def _normalize_diffspec_plugins(plugins):
            names = [
                "salience_encoding",
                "hazard_profile",
                "chunk_arena",
                "bundle_scheduling",
                "paged_kv",
                "fused_kernel",
            ]
            default_flags = {
                "salience_encoding": False,
                "hazard_profile": True,
                "chunk_arena": False,
                "bundle_scheduling": False,
                "paged_kv": False,
                "fused_kernel": False,
            }
            if plugins is None:
                return default_flags.copy()
            if isinstance(plugins, (list, tuple)):
                if len(plugins) != len(names):
                    raise ValueError(f"diffspec_plugins expects {len(names)} flags, got {len(plugins)}")
                return {name: bool(val) for name, val in zip(names, plugins)}
            if isinstance(plugins, dict):
                out = default_flags.copy()
                for name in names:
                    if name in plugins:
                        out[name] = bool(plugins[name])
                return out
            raise TypeError("diffspec_plugins must be list/tuple/dict or None")

        plugin_flags = _normalize_diffspec_plugins(diffspec_plugins)
        if not self.draft_model.use_retrieval_cache:
            plugin_flags = {name: False for name in plugin_flags}
        self.draft_model.diffspec_plugins = plugin_flags
        self.draft_model.diffspec_runtime_policy = {
            "input_len": int(input_len),
            "retrieval_requested": requested_retrieval_cache,
            "retrieval_enabled": bool(self.draft_model.use_retrieval_cache),
            "retrieval_disabled_reason": retrieval_disabled_reason,
            "retrieval_min_context": int(retrieval_min_context) if retrieval_min_context else 0,
            "hybrid_tree_attn_requested": requested_hybrid_tree_attn,
            "hybrid_tree_attn_enabled": bool(self.draft_model.target_use_hybrid_tree_attn),
            "hybrid_tree_attn_disabled_reason": hybrid_disabled_reason,
            "flash_attention_backend": sdpa_backend_name(),
            "external_flash_attn_available": bool(FLASH_ATTN_AVAILABLE),
            "torch_flash_sdp_available": bool(PYTORCH_FLASH_ATTN_AVAILABLE),
            "triton_available": bool(TRITON_AVAILABLE),
            "tree_attention_backend": (
                "triton_hybrid"
                if self.draft_model.target_use_hybrid_tree_attn and TRITON_AVAILABLE
                else "torch_sdpa"
            ),
            "branch_policy": os.environ.get("DIFFSPEC_BRANCH_POLICY", "online"),
            "tree_budget_policy": os.environ.get("DIFFSPEC_TREE_BUDGET_POLICY", "history"),
            "memory_backend": "amd_rocm" if is_rocm_pytorch() else "cuda_or_cpu",
            "amd_residency_hint_requested": _env_flag("DIFFSPEC_AMD_RESIDENCY_HINT", True),
            "plugins": plugin_flags.copy(),
        }
        self.draft_model.enable_salience_encoding = plugin_flags["salience_encoding"]
        self.draft_model.enable_hazard_profile = plugin_flags["hazard_profile"]
        self.draft_model.enable_chunk_arena = plugin_flags["chunk_arena"]
        self.draft_model.enable_bundle_scheduling = plugin_flags["bundle_scheduling"]
        self.draft_model.enable_paged_kv = plugin_flags["paged_kv"]
        self.draft_model.enable_fused_kernel = plugin_flags["fused_kernel"]
        
       
        
        self.draft_model.retrieval_chunk_size = retrieval_chunk_size
        self.draft_model.retrieve_top_k = retrieve_top_k
        self.draft_model.retrieval_verbose = retrieval_verbose
        self.draft_model.retrieve_every_n_steps=retrieve_every_n_steps
        self.draft_model.retrieval_min_context = retrieval_min_context
        self.draft_model.num_chunks_old = 0
        self.draft_model.retrieval_condition = False
        
        self.draft_model.attn_scores = None
        self.draft_model.attn_scores_final = None

        self.draft_model.timestep = 0
        
        # Initialize DiffSpec components (save as model attributes)
        device_str = str(input_ids.device)
        
        # Hazard Profile Tracker for adaptive tree construction
        if self.draft_model.enable_hazard_profile:
            if not hasattr(self, 'hazard_tracker') or self.hazard_tracker is None:
                self.hazard_tracker = HazardProfileTracker(
                    max_depth=max_depth,
                    window_size=50,
                    alpha=0.2,
                    device=device_str
                )
            else:
                self.hazard_tracker.reset()
            self.draft_model.hazard_tracker = self.hazard_tracker
        else:
            self.hazard_tracker = None
            self.draft_model.hazard_tracker = None

        self.tree_budget_controller = HistoryTreeBudgetController.from_env(
            base_nodes=nodes,
            max_depth=max_depth,
        )
        self.draft_model.tree_budget_controller = self.tree_budget_controller
        if self.tree_budget_controller is not None:
            self.draft_model.diffspec_runtime_policy.update(
                {
                    "tree_budget_policy": "late_ramp_history",
                    "tree_budget_window": int(self.tree_budget_controller.window_size),
                    "tree_budget_min_nodes": int(self.tree_budget_controller.min_nodes),
                    "tree_budget_max_nodes": int(self.tree_budget_controller.max_nodes),
                }
            )
        else:
            self.draft_model.diffspec_runtime_policy["tree_budget_policy"] = "fixed"
        
        # Salience-Aware Chunk Encoder
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.draft_model.enable_salience_encoding:
            if not hasattr(self, 'chunk_encoder') or self.chunk_encoder is None:
                self.chunk_encoder = SalienceAwareChunkEncoder(
                    head_dim=head_dim,
                    enable_boundary_preservation=True,
                    enable_learned_projection=True,
                    device=device_str,
                    dtype=torch.float16
                )
            self.draft_model.chunk_encoder = self.chunk_encoder
        else:
            self.chunk_encoder = None
            self.draft_model.chunk_encoder = None
        
        # Chunk Arena for KV cache management (if using retrieval cache)
        if self.draft_model.use_retrieval_cache and self.draft_model.enable_chunk_arena:
            if not hasattr(self, 'chunk_arena') or self.chunk_arena is None:
                kv_num_heads = getattr(self.draft_model.midlayer.self_attn, "num_key_value_heads", None)
                if kv_num_heads is None:
                    kv_num_heads = getattr(self.draft_model.midlayer.self_attn, "num_heads", self.config.num_attention_heads)
                kv_head_dim = getattr(self.draft_model.midlayer.self_attn, "head_dim", head_dim)
                if is_rocm_pytorch():
                    arena_config = AmdArenaConfig(
                        max_chunks=retrieve_top_k,
                        chunk_size=retrieval_chunk_size,
                        num_layers=self.config.num_hidden_layers,
                        num_heads=kv_num_heads,
                        head_dim=kv_head_dim,
                        dtype=torch.float16,
                        device=device_str,
                        enable_residency_hint=_env_flag("DIFFSPEC_AMD_RESIDENCY_HINT", True),
                        residency=AmdResidencyConfig(
                            enable=_env_flag("DIFFSPEC_AMD_RESIDENCY_HINT", True),
                            hit_ratio=float(os.environ.get("DIFFSPEC_AMD_APW_HIT_RATIO", "0.85")),
                            max_window_bytes=(
                                int(os.environ["DIFFSPEC_AMD_APW_MAX_BYTES"])
                                if os.environ.get("DIFFSPEC_AMD_APW_MAX_BYTES")
                                else None
                            ),
                        ),
                    )
                    self.chunk_arena = AmdChunkArena(arena_config)
                    self.draft_model.diffspec_runtime_policy["memory_backend"] = "amd_rocm"
                else:
                    arena_config = ChunkArenaConfig(
                        max_chunks=retrieve_top_k,
                        chunk_size=retrieval_chunk_size,
                        num_layers=self.config.num_hidden_layers,
                        num_heads=kv_num_heads,
                        head_dim=kv_head_dim,
                        dtype=torch.float16,
                        device=device_str,
                        enable_apw=False  # APW requires a CUDA extension path.
                    )
                    self.chunk_arena = ChunkArena(arena_config)
            else:
                self.chunk_arena.reset()
            self.draft_model.chunk_arena = self.chunk_arena
        else:
            self.chunk_arena = None
            self.draft_model.chunk_arena = None

        # Paged KV cache (experimental)
        if self.draft_model.enable_paged_kv:
            # Use KV heads to match KV cache tensor shapes (e.g., GQA uses fewer KV heads).
            paged_num_heads = getattr(self.draft_model.midlayer.self_attn, "num_key_value_heads", None)
            if paged_num_heads is None:
                paged_num_heads = getattr(self.draft_model.midlayer.self_attn, "num_heads", self.config.num_attention_heads)
            paged_head_dim = getattr(self.draft_model.midlayer.self_attn, "head_dim", head_dim)
            paged_config = PagedKVConfig(
                page_size=16,
                max_pages=1024,
                num_layers=self.config.num_hidden_layers,
                num_heads=paged_num_heads,
                head_dim=paged_head_dim,
                dtype=torch.float16,
                device=device_str,
                enable_cow=True,
            )
            self.paged_kv = PagedKVCache(paged_config)
            self.paged_kv.allocate_sequence(seq_id=0, num_tokens=input_len)
            self.draft_model.paged_kv = self.paged_kv
            self.draft_model.paged_kv_seq_id = 0
        else:
            self.paged_kv = None
            self.draft_model.paged_kv = None

        # Bundle scheduler (stats/instrumentation)
        if self.draft_model.enable_bundle_scheduling:
            if not hasattr(self, 'bundle_scheduler') or self.bundle_scheduler is None:
                self.bundle_scheduler = BundleScheduler(
                    bundle_size=3,
                    max_bundles=100,
                    device=device_str
                )
            self.draft_model.bundle_scheduler = self.bundle_scheduler
        else:
            self.bundle_scheduler = None
            self.draft_model.bundle_scheduler = None

        # Fused bundle verifier (experimental)
        if self.draft_model.enable_fused_kernel:
            if not hasattr(self, 'bundle_verifier') or self.bundle_verifier is None:
                self.bundle_verifier = FusedBundleVerifier(
                    bundle_size=3,
                    enable_triton=True,
                    device=device_str
                )
            self.draft_model.bundle_verifier = self.bundle_verifier
        else:
            self.bundle_verifier = None
            self.draft_model.bundle_verifier = None

        input_len = input_ids.shape[1]

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.draft_model.reset_kv()
        
        cache_tree_nodes = (
            self.tree_budget_controller.cache_nodes
            if self.tree_budget_controller is not None
            else nodes
        )
        self.draft_model.full_cache_budget = input_len + max_new_tokens + cache_tree_nodes + max_depth + 16

        # initialize draft model caches
        if self.draft_model.use_retrieval_cache:
            self.init_caches()
        else:
            # self.draft_model.reset_kv()
            self.draft_model.draft_stable_kv = None
            self.draft_model.full_draft_kv = None
            self.draft_model.evicted = 0

        # Initialize target model caches
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(self.base_model, self.draft_model.full_cache_budget)
        self.past_key_values = past_key_values
        self.past_key_values_data = past_key_values_data
        self.current_length_data = current_length_data

        start_time = datetime.now()
        profile_enabled = os.environ.get("DIFFSPEC_PROFILE", "").lower() in {"1", "true", "yes"}
        profile_stats = {
            "initial_setup_time": 0.0,
            "target_tree_decode_time": 0.0,
            "verify_and_next_draft_time": 0.0,
            "iterations": 0,
        }

        def _profile_sync():
            if profile_enabled and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        initial_setup_start = time.perf_counter()
        initial_tree_nodes = (
            self.tree_budget_controller.current_nodes
            if self.tree_budget_controller is not None
            else nodes
        )
        self.draft_model.last_tree_node_budget = int(initial_tree_nodes)
        draft_input_ids,draft_position_ids,tree_attention_mask,last_token,parent=self(input_ids, past_key_values=past_key_values, output_orig=True, nodes=initial_tree_nodes, threshold=threshold, max_depth=max_depth,logits_processor=logits_processor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profile_stats["initial_setup_time"] = time.perf_counter() - initial_setup_start

        draft_input_ids=torch.cat([last_token,draft_input_ids],dim=-1)
        draft_position_ids=torch.cat([torch.tensor([draft_position_ids[0]-1],device=draft_position_ids.device), draft_position_ids],dim=-1)
        tree_attention_mask=torch.cat([torch.zeros(1,tree_attention_mask.size(1),dtype=tree_attention_mask.dtype,device=tree_attention_mask.device),tree_attention_mask],dim=0)
        tree_attention_mask = torch.cat([torch.ones(tree_attention_mask.size(0), 1,dtype=tree_attention_mask.dtype,device=tree_attention_mask.device), tree_attention_mask],
                                        dim=1)
        
        new_token = 0
        total_tokens_list = []
        accept_length_list = []
        gen_time_total = 0
        verify_time_total = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profiler_active = _maybe_cuda_profiler_start("diffspec_decode")
        decode_start = time.perf_counter()
        
        while True:
            assert past_key_values[0][0].shape[2]==draft_position_ids[0]

       
            
            _profile_sync()
            target_decode_start = time.perf_counter()
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_input_ids,
                past_key_values,
                draft_position_ids,
                tree_attention_mask,
            )
            _profile_sync()
            profile_stats["target_tree_decode_time"] += time.perf_counter() - target_decode_start
            

            old_len = input_ids.shape[1]
            
            _profile_sync()
            verify_start = time.perf_counter()
            input_ids,best_candidate,accept_length,draft_input_ids,draft_position_ids,tree_attention_mask,parent=verify(
                                                                      input_ids,
                                                                      logits,
                                                                      draft_input_ids,
                                                                      draft_position_ids,
                                                                      hidden_state_new,
                                                                      tree_attention_mask,
                                                                      past_key_values_data,
                                                                      current_length_data,
                                                                      parent,
                                                                      self,
                                                                      nodes,
                                                                      threshold,
                                                                      max_depth,
                                                                      logits_processor,
                                                                      hazard_tracker=self.hazard_tracker if self.hazard_tracker is not None else None)
            _profile_sync()
            profile_stats["verify_and_next_draft_time"] += time.perf_counter() - verify_start
            profile_stats["iterations"] += 1
            
            

            accept_length_list.append(accept_length.item() if isinstance(accept_length, torch.Tensor) else int(accept_length))

            generated_tokens_list = print_newly_accepted_tokens(old_len, input_ids,
                                                        self.tokenizer, verbose=verbose)

            new_token+=accept_length+1
            total_tokens_list.extend(generated_tokens_list)
           
            
            self.draft_model.timestep += 1

            # if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            #     break
            if new_token > max_new_tokens:
                break

        # Calculate eval metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - decode_start
        _maybe_cuda_profiler_stop(profiler_active)
        avg_accept_length = round(sum(accept_length_list)/len(accept_length_list), 3)
        inference_time = (datetime.now() - start_time).total_seconds()
        total_generated = new_token
        tokens_per_sec = round(total_generated/inference_time,2)
        decode_tokens_per_sec = total_generated / decode_time if decode_time > 0 else 0.0
        
        # Get hazard tracker statistics
        hazard_stats = self.hazard_tracker.get_statistics() if self.hazard_tracker is not None else None

        if output_result_line:
            print(colored(
                f"\nGenerated {total_generated} tokens in {inference_time:.2f}s. "
                f"\nToken/sec: {tokens_per_sec}"
                f"\nAverage acceptance length: {avg_accept_length:.3f}",
                'cyan'
            ))
            if hazard_stats is not None:
                print(colored(
                    f"[Hazard Tracker] Avg reject depth: {hazard_stats['avg_reject_depth']:.2f}, "
                    f"Max hazard at depth {hazard_stats['max_hazard_depth']} ({hazard_stats['max_hazard_prob']:.3f})",
                    'cyan'
                ))
                if verbose:
                    print(colored("\n" + self.hazard_tracker.visualize_profile(), 'yellow'))

        results = {
            'avg_accept_length': avg_accept_length,
            'total_generated': total_generated,
            'inference_time': inference_time,
            'tokens_per_sec': tokens_per_sec,
            'decode_time': decode_time,
            'decode_tokens_per_sec': decode_tokens_per_sec,
            'accept_length_list': accept_length_list,
            'diffspec_plugins': plugin_flags,
            'runtime_policy': self.draft_model.diffspec_runtime_policy,
        }
        if self.tree_budget_controller is not None:
            results['tree_budget_stats'] = self.tree_budget_controller.get_statistics()
        if getattr(self.draft_model, "chunk_arena", None) is not None:
            results['chunk_arena_stats'] = self.draft_model.chunk_arena.get_statistics()
        if hazard_stats is not None:
            results['hazard_stats'] = hazard_stats
        if profile_enabled:
            results['profile_stats'] = profile_stats
        if return_generated_text:
            generated_token_ids = input_ids[0, input_len:].detach().cpu().tolist()
            results['generated_token_ids'] = generated_token_ids
            results['generated_text'] = self.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        
        return results

    # --------- Chunk overlap helpers ---------
    def plot_chunk_overlap_history(self, *args, **kwargs):
        """
        Delegate plotting to the draft model.
        """
        if hasattr(self.draft_model, "plot_chunk_overlap_history"):
            return self.draft_model.plot_chunk_overlap_history(*args, **kwargs)
        raise AttributeError("Underlying draft model does not expose plot_chunk_overlap_history.")

    def save_chunk_overlap_history(self, *args, **kwargs):
        """
        Persist overlap data via the draft model.
        """
        if hasattr(self.draft_model, "save_chunk_overlap_history"):
            return self.draft_model.save_chunk_overlap_history(*args, **kwargs)
        raise AttributeError("Underlying draft model does not expose save_chunk_overlap_history.")

    def reset_chunk_overlap_history(self):
        """
        Clear cached overlap stats so each run can start fresh.
        """
        if hasattr(self.draft_model, "reset_chunk_overlap_history"):
            self.draft_model.reset_chunk_overlap_history()

    
    @torch.no_grad()
    def ar_generate(
            self,
            input_ids: torch.LongTensor,
            max_new_tokens: int = 256,
            eos_token_id: int = None,
            verbose: bool = False,
    ):
        """
        Naive autoregressive generation using a KV cache.

        If verbose=True, prints each newly generated token in green (with proper spacing).
        If verbose=False, returns the full generated string at the end (also properly spaced).

        Args:
            input_ids (torch.LongTensor): shape (1, seq_len), the prompt tokens.
            max_new_tokens (int): maximum number of tokens to append.
            eos_token_id (int, optional): if provided, stop once this token is generated.
            verbose (bool): if True, print each token in green as it's generated.

        Returns:
            If verbose=False: a single string containing prompt + all generated tokens.
            If verbose=True: None (tokens are printed directly with coloring/spacing).
        """
        # Ensure batch size == 1
        assert input_ids.ndim == 2 and input_ids.shape[0] == 1, "batch size >1 is not supported"
        input_len = input_ids.shape[1]
        self.full_cache_budget = input_len + max_new_tokens + 16

        # Build an empty KV cache structure for base_model
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(self.base_model, self.full_cache_budget)
        self.past_key_values = past_key_values
        self.past_key_values_data = past_key_values_data
        self.current_length_data = current_length_data

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Working tensor of generated IDs; start with the prompt.
        generated = input_ids

        # If not verbose, accumulate into this buffer to return later.
        # We decode the *prompt* first.
        full_output = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for step in range(max_new_tokens):
            if step == 0:
                # First forward: feed the entire prompt and request the KV cache.
                outputs = self.base_model(
                    input_ids=generated,
                    use_cache=True,
                    return_kv=True,
                    init=True
                )
            else:
                # On subsequent steps, only feed the last token + past_key_values
                outputs = self.base_model(
                    input_ids=next_token,           # shape (1,1)
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_kv=True
                )

            # Retrieve logits & new KV cache
            logits = outputs.logits

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)

            generated = torch.cat([generated, next_token], dim=1)

            token_str = self.tokenizer.decode(
                next_token.squeeze(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # If the decoded token does not start with a space, prefix one:
            if token_str and not token_str.startswith(" "):
                spaced_token = " " + token_str
            else:
                spaced_token = token_str

            if verbose:
                # Print in green, flush immediately, no newline
                print(colored(spaced_token, 'green'), end="", flush=True)
            else:
                # Accumulate into buffer (which already contains the prompt)
                full_output += spaced_token

            # Stop early if we hit EOS
            if next_token.item() == eos_token_id:
                break

        if not verbose:
            return full_output
        else:
            # Print a final newline for cleanliness
            print()
            return
    
    
    
    @torch.no_grad()
    def autoregressive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=256,
            output_result_line=False,
            verbose=True,
            return_generated_text=False,
            use_flash_prefill=True,
    ):
        """
        Autoregressive baseline generation with one accepted token per decode step.
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_len = input_ids.shape[1]

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
            
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        
        # Initialize target model caches
        cache_budget = input_len + max_new_tokens + 16
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(self.base_model, cache_budget)

        start_time = datetime.now()
        
        # Prefill over the complete prompt.
        if verbose:
            print(colored(f"Starting autoregressive generation with input length {input_len}", 'yellow'))
            
        position_ids = torch.arange(input_len, device=input_ids.device).unsqueeze(0)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.base_model.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                    init=True,
                    target_use_flash_prefill=use_flash_prefill,
                )
            
            # Logits for the first decode token.
            logits = self.base_model.lm_head(outputs.last_hidden_state[:, -1:])
	            
            # Publish the prefill length to the static KV cache wrapper.
            current_length_data.fill_(input_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_time = time.perf_counter() - prefill_start
	        
        new_token = 0
        generated_tokens_list = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profiler_active = _maybe_cuda_profiler_start("auto_decode")
        decode_start = time.perf_counter()
	        
        # Autoregressive decode loop.
        while new_token < max_new_tokens:
            # Sample or greedily select the next token.
            if logits_processor is not None:
                processed_logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(processed_logits, dim=-1)
                next_token = torch.multinomial(probabilities[0], 1).unsqueeze(0)
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            # Stop at EOS.
            if self.tokenizer.eos_token_id in next_token.tolist():
                break
                
            # Append the accepted token.
            next_token = next_token.to(input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            new_token += 1
            
            # Stream tokens only when requested by the caller.
            if verbose:
                token_text = self.tokenizer.decode(next_token[0].tolist(), skip_special_tokens=True)
                generated_tokens_list.append(token_text)
                print(colored(token_text, 'yellow'), end='', flush=True)
            
            # Feed the last token into the next decode step.
            if new_token < max_new_tokens:
                current_pos = current_length_data[0].item()
                position_ids = torch.tensor([[current_pos]], device=input_ids.device)
                
                with torch.inference_mode():
                    outputs = self.base_model.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        use_cache=True
                    )
                    
                    logits = self.base_model.lm_head(outputs.last_hidden_state)
                    
                    # Publish the updated cache length.
                    current_length_data.fill_(current_pos + 1)
           	       
        # Report wall-clock and decode-only throughput.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - decode_start
        _maybe_cuda_profiler_stop(profiler_active)
        inference_time = (datetime.now() - start_time).total_seconds()
        total_generated = new_token
        tokens_per_sec = round(total_generated/inference_time, 2) if inference_time > 0 else 0
        decode_tokens_per_sec = total_generated / decode_time if decode_time > 0 else 0
        
        if output_result_line:
            print(colored(
                f"\nAutoregressive generation completed: generated {total_generated} tokens "
                f"in {inference_time:.2f}s."
                f"\nToken/sec: {tokens_per_sec}",
                'cyan'
            ))

        results = {
            'total_generated': total_generated,
            'inference_time': inference_time,
            'tokens_per_sec': tokens_per_sec,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'decode_tokens_per_sec': decode_tokens_per_sec,
            'method': 'autoregressive',
            'generated_text': ''.join(generated_tokens_list),
            'runtime_policy': {
                'flash_prefill_enabled': bool(use_flash_prefill),
                'flash_attention_backend': sdpa_backend_name(),
                'external_flash_attn_available': bool(FLASH_ATTN_AVAILABLE),
                'torch_flash_sdp_available': bool(PYTORCH_FLASH_ATTN_AVAILABLE),
            },
        }
        if return_generated_text:
            generated_token_ids = input_ids[0, input_len:].detach().cpu().tolist()
            results['generated_token_ids'] = generated_token_ids
            results['generated_text'] = self.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        return results

        
        
    
    
    def init_caches(self):
        # Get values from the model config.
        num_hidden_layers = self.draft_model.config.num_hidden_layers

        # [FIXED] Use actual attention layer configuration, not config
        # The draft model may use GQA (Grouped Query Attention) with fewer key-value heads
        num_heads = self.draft_model.midlayer.self_attn.num_key_value_heads  # Use actual KV heads
        head_dim = self.draft_model.midlayer.self_attn.head_dim
        
        # full cache: shape: [layers, batch_size, num_heads, full_cache_budget, head_dim]
        self.draft_model.full_draft_kv = []
        for layer_idx in range(num_hidden_layers):
            device = self.draft_model.midlayer.self_attn.q_proj.weight.device
            full_K = torch.zeros(
                [1, num_heads, self.draft_model.full_cache_budget, head_dim],
                dtype=torch.float16,
                device=device
            )
            full_V = torch.zeros(
                [1, num_heads, self.draft_model.full_cache_budget, head_dim],
                dtype=torch.float16,
                device=device
            )
            self.draft_model.full_draft_kv.append((full_K, full_V))

        self.draft_model.total_seq_len = 0
        self.draft_model.seq_len_total_old = 0
        self.draft_model.evicted = 0
        self.draft_model.draft_stable_kv = None
        self.draft_model.chunks = None
        self.draft_model.cached_chunks = None
        self.draft_model.selected_chunks = []
        self.draft_model.prev_selected_chunk_ids = []
        self.draft_model.chunk_reps = None
        self.draft_model.past_key_position_ids = None
        
        self.draft_model.recent_start = 0
        self.draft_model.recent_end = 0
        
        
        self.draft_model.chunk_reps = None
        self.draft_model.prev_selected_chunk_ids = []
        # Tensorized chunk descriptors on the draft device.
        self.draft_model._chunks_se = None          # [N, 2] -> (start, end), long, device=self.device
        self.draft_model._chunks_starts = None      # [N]
        self.draft_model._chunks_ends = None        # [N]

        # Scratch buffers for interval-union index construction.
        self.draft_model._diff_buf = None           # [full_cache_budget+1] int32, device=self.device
        self.draft_model._ones_buf = None           # [max_chunks] int32 constant +1
        self.draft_model._minus_ones_buf = None     # [max_chunks] int32 constant -1

        
