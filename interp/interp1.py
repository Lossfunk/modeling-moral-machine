
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moral_evaluation import MoralReasoningTransformer, load_model, CHARACTERS, CHAR_TO_IDX


def _to_scenario_tensor(device: torch.device, scenario_tuple: Tuple[Dict[str, int], Dict[str, int]]) -> torch.Tensor:
    """
    Convert (dict_0, dict_1) -> tensor of shape (1, 2, 23) of dtype long on device.
    """
    outcome_0, outcome_1 = scenario_tuple
    num_chars = len(CHARACTERS)

    vec_0 = np.zeros(num_chars, dtype=np.int64)
    vec_1 = np.zeros(num_chars, dtype=np.int64)

    for char, count in outcome_0.items():
        vec_0[CHAR_TO_IDX[char]] = int(count)
    for char, count in outcome_1.items():
        vec_1[CHAR_TO_IDX[char]] = int(count)

    # (batch=1, outcomes=2, characters=23)
    stacked = np.stack([vec_0, vec_1], axis=0)  # ensure single ndarray to avoid slow tensor creation
    scenario = torch.from_numpy(stacked).unsqueeze(0)  # (1, 2, num_chars)
    scenario = scenario.to(device=device, dtype=torch.long)
    return scenario


def _split_team_token_ranges(num_characters: int) -> Tuple[slice, slice]:
    start_team0 = 0
    end_team0 = num_characters
    start_team1 = end_team0
    end_team1 = start_team1 + num_characters
    return slice(start_team0, end_team0), slice(start_team1, end_team1)


def _format_attn(attn: torch.Tensor, num_heads: Optional[int] = None) -> torch.Tensor:
    if attn.dim() == 4:
        b1, d2, d3, d4 = attn.shape
        if d2 <= 64 and d3 == d4:
            return attn
        if b1 == d3 and d2 < 2048 and d4 == d3:  
            return attn.permute(1, 2, 0, 3).contiguous()  
    elif attn.dim() == 3:
        b, tgt, src = attn.shape
        if num_heads is not None and b % num_heads == 0 and tgt == src:
            real_b = b // num_heads
            return attn.view(real_b, num_heads, tgt, src)
        else:
            return attn.unsqueeze(1)  
    raise RuntimeError(f"Unsupported attention tensor shape: {tuple(attn.shape)}")


def _manual_mha_forward(
    mha: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
        raise NotImplementedError("Bleh")

    batch_first = mha.batch_first
    if batch_first:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    q, k, v = F._in_projection_packed(query, key, value, mha.in_proj_weight, mha.in_proj_bias)

    # Reshape to (batch, heads, seq, head_dim)
    q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)
    k = k.contiguous().view(src_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)
    v = v.contiguous().view(src_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)

    # Scaled dot-product attention
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    if mha.dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=mha.dropout, training=mha.training)

    attn_output = torch.matmul(attn_weights, v)  # (batch, heads, tgt, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)

    # Output projection
    attn_output = F.linear(attn_output, mha.out_proj.weight, mha.out_proj.bias)

    if not batch_first:
        attn_output = attn_output.transpose(0, 1)

    return attn_output, attn_weights


class _MHAAttnTap:
    def __init__(self, model: nn.Module):
        self.model = model
        self.mhas: List[nn.MultiheadAttention] = []
        self.fwd_handles = []
        self._orig_forwards = []
        self.per_layer_attn: List[torch.Tensor] = []
        self.per_layer_attn_grads: List[Optional[torch.Tensor]] = []

        encoder: nn.TransformerEncoder = model.transformer
        for layer in encoder.layers:
            if hasattr(layer, "self_attn"):
                self.mhas.append(layer.self_attn)

        for mha in self.mhas:
            self._patch_mha(mha)
            # attach forward hook to capture attn weights tensors
            handle = mha.register_forward_hook(self._capture_output)
            self.fwd_handles.append(handle)

    def _patch_mha(self, mha: nn.MultiheadAttention):
        orig_forward = mha.forward

        def wrapped_forward(query, key, value, **kwargs):
            attn_mask = kwargs.get("attn_mask")
            key_padding_mask = kwargs.get("key_padding_mask")
            average_attn_weights = kwargs.get("average_attn_weights", False)

            if attn_mask is None and key_padding_mask is None:
                attn_output, attn_weights = _manual_mha_forward(
                    mha, query, key, value
                )
                if average_attn_weights:
                    attn_weights_to_return = attn_weights.mean(dim=1)
                else:
                    attn_weights_to_return = attn_weights
                return attn_output, attn_weights_to_return

            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return orig_forward(query, key, value, **kwargs)

        self._orig_forwards.append((mha, orig_forward))
        mha.forward = wrapped_forward

    def _capture_output(self, module, inputs, output):
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights = output[1]
        else:
            return

        num_heads = module.num_heads if hasattr(module, "num_heads") else None
        A = _format_attn(attn_weights, num_heads=num_heads)

        self.per_layer_attn.append(A)
        self.per_layer_attn_grads.append(None)
        idx = len(self.per_layer_attn) - 1

        if A.requires_grad:
            try:
                A.retain_grad()
            except Exception:
                pass  

            def _save_grad(grad, index=idx, store=self.per_layer_attn_grads):
                store[index] = grad

            A.register_hook(_save_grad)

    def clear(self):
        self.per_layer_attn.clear()
        self.per_layer_attn_grads.clear()

    def remove(self):
        for mha, orig in self._orig_forwards:
            mha.forward = orig
        for h in self.fwd_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._orig_forwards.clear()
        self.fwd_handles.clear()

    def collect_grads(self):
        if len(self.per_layer_attn_grads) < len(self.per_layer_attn):
            missing = len(self.per_layer_attn) - len(self.per_layer_attn_grads)
            self.per_layer_attn_grads.extend([None] * missing)
        return self.per_layer_attn_grads


def _build_identity(batch: int, seq: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    I = torch.eye(seq, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)  # (batch, seq, seq)
    return I


def _batched_matmul(A_list: List[torch.Tensor]) -> torch.Tensor:
    assert len(A_list) > 0
    C = A_list[0]
    for i in range(1, len(A_list)):
        C = torch.matmul(C, A_list[i])
    return C


def attention_rollout_explain(
    model: nn.Module,
    scenario_tensor: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    tap = _MHAAttnTap(model)
    tap.clear()
    _ = model(scenario_tensor)
    layer_A_bars = []
    layer_means = []
    for A in tap.per_layer_attn:
        A = _format_attn(A, num_heads=getattr(tap.mhas[0], "num_heads", None))
        A_mean = A.mean(dim=1)  # Eh A^(b) -> (batch, seq, seq)
        layer_means.append(A_mean)
        I = _build_identity(A_mean.shape[0], A_mean.shape[1], A_mean.device, A_mean.dtype)
        A_bar = I + A_mean
        layer_A_bars.append(A_bar)

    C = _batched_matmul(layer_A_bars)
    tap.remove()
    return C, layer_means


class CheferExplainer:
    def __init__(self, model: nn.Module):
        self.model = model

    def explain_tensor(
        self,
        scenario_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logit: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = scenario_tensor.device
        self.model.zero_grad()
        self.model.eval()
        tap = _MHAAttnTap(self.model)
        tap.clear()
        out = self.model(scenario_tensor)  # (batch=1, 1)
        scalar = out.view(-1)[0]
        if not use_logit:
            scalar = torch.sigmoid(scalar)

        if target is not None:
            scalar = target  

        scalar.backward(retain_graph=True)

        grads = tap.collect_grads()
        A_list = tap.per_layer_attn

        per_layer_Abar = []
        per_layer_metric = []

        for A, G in zip(A_list, grads):
            A = _format_attn(A, num_heads=getattr(tap.mhas[0], "num_heads", None))
            if G is None:
                raise RuntimeError("No gradient on attention weights — ensure backward() was called on a scalar output.")
            G = _format_attn(G, num_heads=A.shape[1])

            M = (G * A)
            # Positive part only
            M = torch.clamp(M, min=0.0)  # (batch, heads, seq, seq)

            # Mean over heads
            M_mean = M.mean(dim=1)  # (batch, seq, seq)
            per_layer_metric.append(M_mean)

            # Add identity to account for residual connection (skip) per Eq. 13
            I = _build_identity(M_mean.shape[0], M_mean.shape[1], device=M_mean.device, dtype=M_mean.dtype)
            A_bar = I + M_mean
            per_layer_Abar.append(A_bar)

        C = _batched_matmul(per_layer_Abar)  # (batch, seq, seq)
        cls_row = C[:, 0, :]  # (batch, seq)

        tap.remove()
        return {"C": C, "cls_row": cls_row, "per_layer_Abar": per_layer_Abar, "per_layer_metric": per_layer_metric}

    def explain_scenario(
        self,
        scenario_tuple: Tuple[Dict[str, int], Dict[str, int]],
        device: Optional[torch.device] = None,
        symmetric: bool = False,
        use_logit: bool = True
    ) -> Dict[str, Dict[str, float]]:
        if device is None:
            device = next(self.model.parameters()).device
        def _single_dir(scenario_t):
            res = self.explain_tensor(scenario_t, use_logit=use_logit)
            cls_row = res["cls_row"][0]  # (seq,)
            token_scores = cls_row[1:]  # (46,)
            denom = token_scores.sum().clamp_min(1e-12)
            token_scores = token_scores / denom
            num_chars = len(CHARACTERS)
            team0_slice, team1_slice = _split_team_token_ranges(num_chars)

            team0_scores = token_scores[team0_slice]
            team1_scores = token_scores[team1_slice]

            team0 = {CHARACTERS[i]: float(team0_scores[i]) for i in range(num_chars)}
            team1 = {CHARACTERS[i]: float(team1_scores[i]) for i in range(num_chars)}
            return team0, team1

        scenario_t1 = _to_scenario_tensor(device, scenario_tuple)

        if not symmetric:
            t0, t1 = _single_dir(scenario_t1)
            return {"team0": t0, "team1": t1}

        # symmetric: also compute flipped scenario and remap
        outcome0, outcome1 = scenario_tuple
        scenario_t2 = _to_scenario_tensor(device, (outcome1, outcome0))

        t0_a, t1_a = _single_dir(scenario_t1)  # original order
        t0_b, t1_b = _single_dir(scenario_t2)  # flipped order

        # For the flipped pass, swap back teams so both are in the original reference frame
        fused_team0 = {}
        fused_team1 = {}
        for i, name in enumerate(CHARACTERS):
            fused_team0[name] = float((torch.tensor(t0_a[name]) + torch.tensor(t1_b[name])) / 2.0)
            fused_team1[name] = float((torch.tensor(t1_a[name]) + torch.tensor(t0_b[name])) / 2.0)

        return {"team0": fused_team0, "team1": fused_team1}


def summarize_relevance(
    relevance: Dict[str, Dict[str, float]],
    counts: Tuple[Dict[str, int], Dict[str, int]]
) -> Dict[str, List[Tuple[str, float]]]:
    out = {}
    for team_key, team_counts in zip(["team0", "team1"], counts):
        scores = relevance[team_key]
        present = [(c, scores[c]) for c, k in team_counts.items() if k > 0]
        present.sort(key=lambda x: x[1], reverse=True)
        out[team_key] = present
    return out


def _print_relevance(
    relevance: Dict[str, Dict[str, float]],
    scenario_tuple: Tuple[Dict[str, int], Dict[str, int]],
    mode: str = "present",
    topk: int = 10
) -> None:
    normalized_mode = mode.lower()
    if normalized_mode not in {"present", "all", "topk"}:
        raise ValueError("mode must be one of {'present', 'all', 'topk'}")

    for team_key, outcome in zip(["team0", "team1"], scenario_tuple):
        print(f"\n{team_key.upper()}")

        if normalized_mode == "present":
            items = [(c, relevance[team_key][c]) for c, count in outcome.items() if count > 0]
        elif normalized_mode == "all":
            items = [(c, relevance[team_key][c]) for c in CHARACTERS]
        else:  # normalized_mode == "topk"
            items = sorted(
                [(c, relevance[team_key][c]) for c in CHARACTERS],
                key=lambda x: x[1],
                reverse=True
            )[:topk]

        items.sort(key=lambda x: x[1], reverse=True)
        for name, val in items:
            print(f"  {name:20s}: {val:.4f}")


def explain_and_print(
    model: nn.Module,
    device: torch.device,
    scenario_tuple: Tuple[Dict[str, int], Dict[str, int]],
    symmetric: bool = True,
    use_rollout_baseline: bool = False
) -> None:
    explainer = CheferExplainer(model)
    rel = explainer.explain_scenario(scenario_tuple, device=device, symmetric=symmetric, use_logit=True)
    print("\\n=== Chefer-style explanation (class-specific) ===")
    _print_relevance(rel, scenario_tuple, mode="all")

    if use_rollout_baseline:
        print("\\n=== Attention rollout (class-agnostic baseline) ===")
        scenario_t = _to_scenario_tensor(device, scenario_tuple)
        C_roll, _ = attention_rollout_explain(model, scenario_t)
        cls_row = C_roll[0, 0, 1:]  # drop CLS
        cls_row = cls_row / cls_row.sum().clamp_min(1e-12)
        num_chars = len(CHARACTERS)
        t0_slice, t1_slice = _split_team_token_ranges(num_chars)
        t0 = cls_row[t0_slice].tolist()
        t1 = cls_row[t1_slice].tolist()

        rollout_rel = {
            "team0": {c: t0[i] for i, c in enumerate(CHARACTERS)},
            "team1": {c: t1[i] for i, c in enumerate(CHARACTERS)},
        }
        _print_relevance(rollout_rel, scenario_tuple, mode="all")


if __name__ == "__main__":
    try:
        model, device = load_model("best_model.pt")
        model.eval()
        scenario = ({"Man": 3}, {"Criminal": 3})
        explain_and_print(model, device, scenario, symmetric=True, use_rollout_baseline=True)
    except FileNotFoundError:
        print("best_model.pt not found; import the module and run with your own model.")
    except Exception as e:
        print("Encountered an error in the demo:", repr(e))
