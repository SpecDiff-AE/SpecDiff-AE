import torch

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from termcolor import colored
from ..core.bundle_scheduler import BundleScheduler


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list
    return None


def tree_decoding(
        model,
        draft_input_ids,
        past_key_values,
        draft_position_ids,
        tree_attention_mask,
):
    if model.draft_model.use_retrieval_cache:
        retrieval_condition = model.draft_model.timestep % model.draft_model.retrieve_every_n_steps == 0
        model.draft_model.retrieval_condition = retrieval_condition
    else:
        retrieval_condition = False
    
    outputs, tree_logits, hidden_state = model(
        draft_input_ids,
        tree_attention_mask=tree_attention_mask,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=draft_position_ids,
        init=False,
        retrieve_attn_scores = retrieval_condition,
        best=model.draft_model.best
    )
    
    draft_device = model.draft_model.lm_head.weight.device
    if outputs["hidden_states"][0].device != draft_device:
        outputs["hidden_states"] = [x.to(draft_device) for x in outputs["hidden_states"]]
    hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    return tree_logits, hidden_state, outputs

def verify(input_ids,logits,draft,position_ids,hidden_states,tree_attention_mask,past_key_values_data,current_length_data,parent,model,nodes,threshold,max_depth,logits_processor,hazard_tracker=None):
    """
    Redirects to the optimized verify_new implementation with hazard tracker support.
    """
    return verify_new(input_ids,logits,draft,position_ids,hidden_states,tree_attention_mask,
                     past_key_values_data,current_length_data,parent,model,nodes,threshold,
                     max_depth,logits_processor,hazard_tracker)


@torch.no_grad()
def verify_new(
    input_ids,
    logits,
    draft,
    position_ids,
    hidden_states,
    tree_attention_mask,
    past_key_values_data,
    current_length_data,
    parent,
    model,
    nodes,
    threshold,
    max_depth,
    logits_processor,
    hazard_tracker=None
):
    """
    Verify the selected draft tree and advance the generation state.

    The implementation preserves the public return contract while reducing
    redundant sampling and ancestor-consistency work.
    """

    # 1. Predict one target token for each node in the current draft tree.
    if logits_processor is None:
        next_tokens = torch.argmax(logits, dim=-1)  # [1, M]
    else:
        logits = logits_processor(None, logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]      # [M, V]
        next_tokens = torch.multinomial(probs, 1).view(1, -1)       # [1, M]

    next_tokens = next_tokens.to(draft.device)

    # 2. Select the deepest acceptable node. The tree is small, so doing the
    # ancestor closure on CPU avoids many tiny GPU kernels and sync points.
    parent_cpu = parent.detach().to("cpu")
    draft_cpu = draft[0].detach().to("cpu")
    next_cpu = next_tokens[0].detach().to("cpu")
    if position_ids.dim() == 2:
        pos_cpu = position_ids[0].detach().to("cpu")
    else:
        pos_cpu = position_ids.detach().to("cpu")

    rows_cpu = torch.arange(parent_cpu.numel(), dtype=parent_cpu.dtype)
    parent_cpu = torch.where(parent_cpu == rows_cpu, torch.full_like(parent_cpu, -1), parent_cpu)
    parent_ext_cpu = torch.cat(
        [torch.zeros(1, dtype=parent_cpu.dtype), parent_cpu + 1],
        dim=-1
    )

    expected_cpu = next_cpu.index_select(0, parent_ext_cpu)
    ok_cpu = draft_cpu.eq(expected_cpu)
    ok_cpu[0] = True

    # A child is acceptable only when every ancestor is also accepted.
    max_hops = min(int(max_depth) + 2, int(parent_ext_cpu.numel()))
    for _ in range(max_hops):
        next_ok_cpu = ok_cpu & ok_cpu.index_select(0, parent_ext_cpu)
        if torch.equal(next_ok_cpu, ok_cpu):
            break
        ok_cpu = next_ok_cpu

    if getattr(model.draft_model, "enable_bundle_scheduling", False):
        scheduler = getattr(model.draft_model, "bundle_scheduler", None)
        if scheduler is None:
            scheduler = BundleScheduler(bundle_size=3, max_bundles=100, device=str(draft.device))
            model.draft_model.bundle_scheduler = scheduler
        parent_map = {i: int(parent_ext_cpu[i].item()) for i in range(parent_ext_cpu.numel())}
        parent_map[0] = -1
        children = set(parent_map.values()) - {-1}
        leaf_ids = [i for i in range(parent_ext_cpu.numel()) if i not in children]
        depth_map = {i: int(pos_cpu[i].item()) for i in range(pos_cpu.numel())}
        tree_descriptor = {
            "leaf_ids": leaf_ids,
            "parent_map": parent_map,
            "depth_map": depth_map,
        }
        bundles = scheduler.schedule_bundles(tree_descriptor, return_tensors=False)
        model.draft_model.last_bundle_stats = {
            "num_leaves": len(leaf_ids),
            "num_bundles": len(bundles),
            "avg_prefix_length": scheduler.avg_prefix_length,
        }

    masked_pos_cpu = torch.where(ok_cpu, pos_cpu, torch.zeros_like(pos_cpu))
    max_id = int(torch.argmax(masked_pos_cpu).item())
    
    # Recover the root-to-leaf path for the selected node.
    best_candidate = []
    best_candidate_id = []

    # Treat the root as parentless during traceback.
    parent_back = parent_ext_cpu.tolist()
    parent_back[0] = -1

    cur = int(max_id)
    while cur != -1:
        best_candidate.append(int(draft_cpu[cur].item()))
        best_candidate_id.append(cur)
        cur = int(parent_back[cur])

    best_candidate.reverse()
    best_candidate_id.reverse()

    next_token = next_tokens[0, max_id].view(1, 1)                  # [1,1]
    accept_length = len(best_candidate) - 1

    if hazard_tracker is not None:
        observed_tree_depth = int(max(0, (pos_cpu.max() - pos_cpu[0]).item()))
        # Reviewer-facing hazard: where the selected verification path stops.
        # accept_length=N means depths [0, N-1] were accepted; failure is depth N.
        first_reject_depth = -1 if accept_length >= observed_tree_depth else int(accept_length)
        hazard_tracker.update(first_reject_depth)

    budget_controller = getattr(model, "tree_budget_controller", None)
    next_nodes = int(nodes)
    if budget_controller is not None:
        next_nodes = budget_controller.observe(int(accept_length))
    model.draft_model.last_tree_node_budget = next_nodes

    # 3. Compact selected KV rows and update cache length.
    start = int(current_length_data[0].item()) - int(draft.size(1))
    select_indices = torch.tensor(best_candidate_id, device=input_ids.device, dtype=torch.long) + start

    if getattr(model.draft_model, "use_retrieval_cache", False):
        if getattr(model.draft_model, "retrieval_condition", False):
            prev_input_len = input_ids.shape[1]
            last_query_index = best_candidate_id[-1]
            last_attn_scores = model.draft_model.attn_scores[:, :, last_query_index, :].mean(dim=1).squeeze()
            best_candidate_id_abs = torch.tensor(best_candidate_id, device=last_attn_scores.device) + prev_input_len
            model.draft_model.attn_scores_final = torch.cat(
                (last_attn_scores[:prev_input_len], last_attn_scores[best_candidate_id_abs]),
                dim=0
            )

    for data in past_key_values_data:
        tgt = data[..., select_indices.to(data.device), :]          # (..., S_sel, D)
        dst = data[..., start: start + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)

    current_length_data.fill_(start + tgt.shape[-2])

    # 4. Append the accepted path and build the next draft tree.
    input_ids = torch.cat(
        [input_ids, torch.tensor(best_candidate, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)],
        dim=-1
    )

    new_accept_hidden = hidden_states[:, torch.tensor(best_candidate_id, device=hidden_states.device), :]

    next_draft, next_position_ids, next_tree_attention_mask, next_parent = model.draft_model.build_speculative_tree(
        new_accept_hidden,
        input_ids=torch.cat((input_ids, next_token.to(input_ids.device)), dim=1),
        head=model.base_model.lm_head,
        nodes=next_nodes,
        threshold=threshold,
        max_depth=max_depth
    )

    next_draft = torch.cat([next_token, next_draft], dim=-1)

    if position_ids.dim() == 2:
        root_pos_val = (position_ids[0, max_id] + 1).to(next_position_ids.device)
    else:
        root_pos_val = (position_ids[max_id] + 1).to(next_position_ids.device)
    next_position_ids = torch.cat([root_pos_val.view(1), next_position_ids], dim=-1)

    next_tree_attention_mask = torch.cat(
        [torch.zeros(1, next_tree_attention_mask.size(1), dtype=next_tree_attention_mask.dtype,
                     device=next_tree_attention_mask.device),
         next_tree_attention_mask],
        dim=0
    )
    next_tree_attention_mask = torch.cat(
        [torch.ones(next_tree_attention_mask.size(0), 1, dtype=next_tree_attention_mask.dtype,
                    device=next_tree_attention_mask.device),
         next_tree_attention_mask],
        dim=1
    )

    return input_ids, best_candidate, accept_length, next_draft, next_position_ids, next_tree_attention_mask, next_parent




def print_newly_accepted_tokens(
    old_len,
    input_ids_after,
    tokenizer,
    accepted_color='green',
    resampled_color='blue',
    verbose=False,
):
    """
    Print tokens appended to 'input_ids_after' from index 'old_len' onward.
    
    According to the new logic:
      - The *first* of these newly appended tokens is actually "resampled".
      - Any remaining tokens are "accepted".
    
    We no longer have "rejected" tokens in 'input_ids_after'.
    """
    if not verbose:
        return []

    # Slice the newly appended tokens
    new_tokens = input_ids_after[0, old_len:]
    if new_tokens.numel() == 0:
        return []

    typed_tokens = []

    # 1) The first newly appended token is "resampled"
    resampled_token = new_tokens[0]
    resampled_str = tokenizer.decode(
        resampled_token.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=True
    ).replace("<0x0A>", "\n")
    typed_tokens.append((resampled_str, "resampled"))

    # 2) All remaining newly appended tokens are "accepted"
    if new_tokens.numel() > 1:
        accepted_tokens = new_tokens[1:]
        for t in accepted_tokens:
            decoded = tokenizer.decode(
                t.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).replace("<0x0A>", "\n")
            typed_tokens.append((decoded, "accepted"))

    # 3) Print them with color if verbose
    if verbose:
        for (token_str, token_type) in typed_tokens:
            if token_type == "accepted":
                clr = accepted_color
            else:  # token_type == "resampled"
                clr = resampled_color
            print(colored(token_str, clr), flush=True, end="")

    return typed_tokens
