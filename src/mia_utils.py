from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math
import re
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from joblib import Parallel, delayed


def compute_min_kpp_token_scores(
    record: Dict[str, torch.Tensor],
    stream: str = "chosen",
    *,
    eps_sigma: float = 1e-5,
) -> torch.Tensor:
    """
    Min-K%++ per-token z-score feature.
      s_t = (logprob_true[t] - mu[t]) / max(sigma[t], eps_sigma)

    Shapes (per sample/record)
    - record[f"logprob_{stream}"]: 1-D tensor [T]
    - record[f"mu_{stream}"]     : 1-D tensor [T]
    - record[f"sigma_{stream}"]  : 1-D tensor [T]
    - return                      : 1-D tensor [T]
    """
    # Clamp sigma to avoid exploding scores when variance is near zero.
    sigma_safe = torch.clamp(record[f"sigma_{stream}"], min=eps_sigma)
    return (record[f"logprob_{stream}"] - record[f"mu_{stream}"]) / sigma_safe


def compute_dpo_loss(record: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Sequence-level DPO loss from chosen vs rejected responses:
      L = -logsigmoid( mean_logp(chosen) - mean_logp(rejected) )

    Shapes (per sample/record)
    - record["logprob_chosen"]  : 1-D tensor [Tc]
    - record["logprob_rejected"]: 1-D tensor [Tr]
    - return                   : scalar tensor [] (caller reshapes to [1])
    """
    # Positive delta means chosen is favored by the model.
    delta = record["logprob_chosen"].mean(-1) - record["logprob_rejected"].mean(-1)
    return -F.logsigmoid(delta)


def compute_ipo_loss(
    record: Dict[str, torch.Tensor], beta: float = 0.1
) -> torch.Tensor:
    """
    chosen_logps:   1-D [Tc] when called from compute_feature_vector_for_shadow()
                   (historically could be batched [B, Tc])
    rejected_logps: 1-D [Tr] when called from compute_feature_vector_for_shadow()
                   (historically could be batched [B, Tr])
    beta:           τ; target margin is 1/beta

    Returns a scalar loss [] (caller reshapes to [1]).
    """
    chosen_logps = record["logprob_chosen"]
    rejected_logps = record["logprob_rejected"]
    # h is the sequence-level chosen-vs-rejected margin.
    h = chosen_logps.mean(dim=-1) - rejected_logps.mean(dim=-1)  # (B,)
    loss = 0.5 * ((h - 1.0 / beta) ** 2 + h**2)  # symmetric IPO
    return loss.mean().unsqueeze(0)

def compute_kto_loss(
    record: Dict[str, torch.Tensor],
    beta: float = 0.1,
    z0: float = 0.0,
    lambda_desirable: float = 1.0,
    lambda_undesirable: float = 1.0,
) -> torch.Tensor:
    """
    KTO-style loss on a preference tuple (chosen as desirable, rejected as undesirable).

    We use the default loss-neutral setting (lambda_D=lambda_U=1) and z0=0
    as a practical single-record approximation:
      L = 0.5 * [(1 - sigmoid(beta * (r_chosen - z0))) +
                 (1 - sigmoid(beta * (z0 - r_rejected)))]
    where r_* are sequence-level log-probability scores.
    """
    # Approximate reward terms with sequence-average log-probabilities.
    chosen_reward = record["logprob_chosen"].mean(dim=-1)
    rejected_reward = record["logprob_rejected"].mean(dim=-1)

    z0_t = torch.as_tensor(z0, dtype=chosen_reward.dtype, device=chosen_reward.device)
    chosen_loss = lambda_desirable * (1.0 - torch.sigmoid(beta * (chosen_reward - z0_t)))
    rejected_loss = lambda_undesirable * (1.0 - torch.sigmoid(beta * (z0_t - rejected_reward)))
    loss = 0.5 * (chosen_loss + rejected_loss)
    return loss.mean().unsqueeze(0)

def compute_orpo_loss(
    record: Dict[str, torch.Tensor],
    beta: float = 0.1,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    ORPO objective on a preference tuple:
      L = L_SFT + beta * L_OR
    where
      L_SFT = -mean_logp(chosen)
      L_OR  = -logsigmoid( log_odds(chosen) - log_odds(rejected) )
    with log_odds(y) = log P(y|x) - log(1 - P(y|x)).
    """
    # Use sequence-level average log-probability as a proxy for P(y|x).
    chosen_logp = record["logprob_chosen"].mean(dim=-1)
    rejected_logp = record["logprob_rejected"].mean(dim=-1)

    chosen_p = chosen_logp.exp().clamp(min=eps, max=1.0 - eps)
    rejected_p = rejected_logp.exp().clamp(min=eps, max=1.0 - eps)

    chosen_logodds = torch.log(chosen_p) - torch.log1p(-chosen_p)
    rejected_logodds = torch.log(rejected_p) - torch.log1p(-rejected_p)

    sft_loss = -chosen_logp
    or_loss = -F.logsigmoid(chosen_logodds - rejected_logodds)
    loss = sft_loss + beta * or_loss
    return loss.mean().unsqueeze(0)


import torch
import torch.nn.functional as F
from typing import Dict


def compute_dpo_logprob(record: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    RMIA-ready DPO log-mass for a single record.

    Returns:
      V = log σ( mean_logp(chosen) - mean_logp(rejected) )
        = - compute_dpo_loss(record)
    Shape: scalar tensor [] (caller may reshape to [1]).
    """
    return -compute_dpo_loss(record)


def compute_ipo_logprob(record: Dict[str, torch.Tensor], beta: float = 0.1) -> torch.Tensor:
    """
    RMIA-ready IPO log-mass for a single record.

    We treat IPO as an energy model: q(z|θ) ∝ exp( - L_IPO(z) ).
    Returns:
      V = - L_IPO(z)
        = - compute_ipo_loss(record, beta)   (with scalar-safe squeeze)
    Shape: scalar tensor [].
    """
    # Older compute_ipo_loss returns [1] in your snippet; squeeze to [] for consistency.
    return -compute_ipo_loss(record, beta=beta).squeeze(0)


def compute_kto_logprob(
    record: Dict[str, torch.Tensor],
    beta: float = 0.1,
    z0: float = 0.0,
    lambda_desirable: float = 1.0,
    lambda_undesirable: float = 1.0,
) -> torch.Tensor:
    """
    RMIA-ready KTO-proxy log-mass for a single record.

    This follows your current KTO-style proxy: q(z|θ) ∝ exp( - L_KTO_proxy(z) ).
    Returns:
      V = - L_KTO_proxy(z)
        = - compute_kto_loss(...)
    Shape: scalar tensor [].
    """
    return -compute_kto_loss(
        record,
        beta=beta,
        z0=z0,
        lambda_desirable=lambda_desirable,
        lambda_undesirable=lambda_undesirable,
    ).squeeze(0)


def compute_orpo_logprob(
    record: Dict[str, torch.Tensor],
    beta: float = 0.1,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    RMIA-ready ORPO log-mass for a single record.

    Returns:
      V = - L_ORPO(z)
        = - compute_orpo_loss(...)
    Shape: scalar tensor [].
    """
    return -compute_orpo_loss(record, beta=beta, eps=eps).squeeze(0)


def decide_fixed_D(
    max_seq_len: int,
    k_pct: float,
    fixed_d: Optional[int] = None,
) -> int:
    """
    Decide the fixed feature length D from either a user override (fixed_d)
    or from k_pct of a global sequence length cap (max_seq_len).
    """
    if fixed_d is not None and fixed_d > 0:
        return int(fixed_d)
    k = max(1, math.ceil((k_pct / 100.0) * int(max_seq_len)))
    return int(k)


def _parse_grouped_truncation_strategy(
    truncation_strategy: str,
) -> Optional[Tuple[str, int]]:
    """
    Parse grouped reducers:
      average_<X>, min_<X>, max_<X>  where X is a positive integer.
    """
    if truncation_strategy is None:
        return None
    m = re.fullmatch(
        r"(average|min|max)_(\d+)",
        str(truncation_strategy).strip().lower(),
    )
    if m is None:
        return None
    group_size = int(m.group(2))
    if group_size <= 0:
        raise ValueError(
            f"Invalid grouped truncation_strategy '{truncation_strategy}': X must be > 0."
        )
    return m.group(1), group_size


def _pad_or_tail_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    x = x.reshape(-1)
    if target_len <= 0:
        return x.new_zeros((0,))
    if x.numel() < target_len:
        return F.pad(x, (0, target_len - x.numel()), value=0.0)
    return x[..., -target_len:]


def _group_reduce_1d(
    scores_1d: torch.Tensor,
    *,
    group_size: int,
    reducer: str,
) -> torch.Tensor:
    """
    Reduce contiguous token groups of size `group_size` with reducer in
    {"average", "min", "max"}.
    """
    x = scores_1d.reshape(-1)
    if x.numel() == 0:
        return x.new_zeros((0,))

    n = int(x.numel())
    n_groups = (n + group_size - 1) // group_size
    pad = n_groups * group_size - n
    if pad > 0:
        x = torch.cat(
            [
                x,
                torch.full((pad,), float("nan"), dtype=x.dtype, device=x.device),
            ],
            dim=0,
        )
    x = x.view(n_groups, group_size)
    finite = torch.isfinite(x)

    if reducer == "average":
        counts = finite.sum(dim=1).clamp_min(1).to(dtype=x.dtype)
        sums = torch.where(finite, x, torch.zeros_like(x)).sum(dim=1)
        return sums / counts
    if reducer == "min":
        mins = torch.where(finite, x, torch.full_like(x, float("inf"))).min(dim=1).values
        return torch.where(torch.isfinite(mins), mins, torch.zeros_like(mins))
    if reducer == "max":
        maxs = torch.where(finite, x, torch.full_like(x, float("-inf"))).max(dim=1).values
        return torch.where(torch.isfinite(maxs), maxs, torch.zeros_like(maxs))

    raise ValueError(f"Unknown grouped reducer: {reducer}")


def build_fixed_length_vector(
    scores_1d: torch.Tensor,
    *,
    D: int,
    truncation_strategy: str = "tail",
) -> Optional[torch.Tensor]:
    """
    Convert variable-length per-token scores into a fixed-length vector.

    Input shapes
    - scores_1d: 1-D tensor [T] for stream in {"chosen","rejected"}
    - scores_1d: tuple(t_chosen, t_rejected) for stream="both", each 1-D

    Output shapes
    - truncation_strategy="tail": [D] (for stream="both", concatenates chosen||rejected then tails/pads)
    - truncation_strategy="average": [1] (single stream) or [2] (both streams: [mean_chosen, mean_rejected])
    - truncation_strategy="half_pad_concat": [D] (both streams: pad/tail each side separately then concatenate)
    - truncation_strategy in {"average_<X>", "min_<X>", "max_<X>"}:
      reduce contiguous groups of X tokens first; then:
      - single stream -> pad/tail to [D]
      - both streams  -> reduce each side separately, pad/tail to half lengths, concat to [D]

    Note: despite the Optional return type, this function currently always returns a tensor
    (it pads with zeros if the sequence is shorter than D).
    """
    if isinstance(scores_1d, tuple):
        scores_1d = (scores_1d[0].reshape(-1), scores_1d[1].reshape(-1))
    else:
        scores_1d = scores_1d.reshape(-1)

    grouped = _parse_grouped_truncation_strategy(truncation_strategy)
    if grouped is not None:
        reducer, group_size = grouped
        if isinstance(scores_1d, tuple):
            d_chosen = int(D // 2)
            d_rejected = int(D - d_chosen)
            chosen_scores = _group_reduce_1d(
                scores_1d[0], group_size=group_size, reducer=reducer
            )
            rejected_scores = _group_reduce_1d(
                scores_1d[1], group_size=group_size, reducer=reducer
            )
            chosen_fixed = _pad_or_tail_1d(chosen_scores, d_chosen)
            rejected_fixed = _pad_or_tail_1d(rejected_scores, d_rejected)
            return torch.cat([chosen_fixed, rejected_fixed], dim=0)
        grouped_scores = _group_reduce_1d(
            scores_1d, group_size=group_size, reducer=reducer
        )
        return _pad_or_tail_1d(grouped_scores, D)

    if truncation_strategy == "average":
        if isinstance(scores_1d, tuple):  # both streams
            return torch.tensor([scores_1d[0].mean(), scores_1d[1].mean()])
        else:
            return scores_1d.mean(-1, keepdim=True)
    elif truncation_strategy == "tail":
        # For stream='both', concatenate chosen and rejected token scores first.
        if isinstance(scores_1d, tuple):
            scores_1d = torch.cat([scores_1d[0], scores_1d[1]], dim=0)

        return _pad_or_tail_1d(scores_1d, D)
    elif truncation_strategy == "half_pad_concat":
        # For stream='both': pad/truncate chosen and rejected separately to
        # half lengths, then concatenate.
        if isinstance(scores_1d, tuple):
            d_chosen = int(D // 2)
            d_rejected = int(D - d_chosen)
            chosen_scores = scores_1d[0]
            rejected_scores = scores_1d[1]

            chosen_fixed = _pad_or_tail_1d(chosen_scores, d_chosen)
            rejected_fixed = _pad_or_tail_1d(rejected_scores, d_rejected)

            return torch.cat([chosen_fixed, rejected_fixed], dim=0)

        # For single-stream modes, keep behavior aligned with tail.
        return _pad_or_tail_1d(scores_1d, D)
    else:
        raise ValueError(f"Unknown truncation_strategy: {truncation_strategy}")


def _to1d_float_cpu(x) -> torch.Tensor:
    """Coerce list/np/torch to 1-D torch.float64 on CPU."""
    t = torch.as_tensor(x)
    if t.dim() == 0:
        t = t.reshape(1)
    return t.to(dtype=torch.float64, device="cpu").reshape(-1)


def _iter_loss_records(loss_records: Dict[str, Any]):
    """
    Yields (sample_id:int, lp_1d, mu_1d, sigma_1d) for each entry in loss_records.
    Handles both list-of-tensors and stacked tensors.
    Expected keys: 'index', f'logprob_{stream}', f'mu_{stream}', f'sigma_{stream}'

    Output record format (per sample)
    - record["index"] is an int sample id (0..N-1)
    - record["logprob_chosen"] is a 1-D torch.float64 CPU tensor [T]
    - record["logprob_both"] (if present) is a tuple: (logprob_chosen[Tc], logprob_rejected[Tr])
    """
    new_records = {}
    idx = loss_records["index"]

    # Normalize index to list[int]
    if isinstance(idx, torch.Tensor):
        idx_list = [int(i) for i in idx.reshape(-1).tolist()]
    elif isinstance(idx, (list, tuple)):
        idx_list = [int(i) for i in idx]
    else:
        raise TypeError(f"Unsupported index type: {type(idx)}")
    new_records["index"] = idx_list
    new_records.update(
        {
            k: [_to1d_float_cpu(a) for a in loss_records[k]]
            for k in loss_records
            if k not in ["index", "is_member"]
        }
    )

    # for rejected in list(new_records.keys()):
    #     if 'rejected' in rejected and rejected not in ['sigma_rejected']:
    #         new_records[rejected] = [-a for a in new_records[rejected]]

    for rejected in list(new_records.keys()):
        if "rejected" in rejected:
            chosen = rejected.replace("rejected", "chosen")
            if chosen in new_records:
                both = rejected.replace("rejected", "both")
                # Convenience for stream="both": if e.g. "logprob_chosen" and
                # "logprob_rejected" exist, also expose "logprob_both" whose
                # value is a 2-tuple (chosen_1d, rejected_1d) for each sample.
                #
                # Downstream code can then do:
                #   scores = record[f"logprob_{stream}"]  # stream can be "both"
                # and pass this tuple into build_fixed_length_vector(), which
                # knows how to handle (chosen, rejected) under different
                # truncation_strategy settings.
                new_records[both] = [
                    (a, b) for a, b in zip(new_records[chosen], new_records[rejected])
                ]

    for i in range(len(idx_list)):
        yield {k: new_records[k][i] for k in new_records}


def merge_streams(
    func: Callable[[Dict[str, Any], str], torch.Tensor],
    record: Dict[str, Any],
    stream: str,
):
    """
    Helper for per-token features where the computation needs the stream string.

    - stream="chosen"/"rejected": returns a 1-D tensor [T]
    - stream="both": returns a tuple(t_chosen[Tc], t_rejected[Tr])
    """
    if stream == "both":
        scores_chosen = func(record, stream="chosen")
        scores_rejected = func(record, stream="rejected")
        return (scores_chosen, scores_rejected)
    else:
        return func(record, stream=stream)


def compute_premia_loss(record):
    """
    PREMIA-style simple margin score:
      score = mean_logp(chosen) - mean_logp(rejected)
    """
    # Unlike the other methods here, this is a direct score, not a loss.
    return record["logprob_chosen"].mean(-1) - record["logprob_rejected"].mean(-1)

def compute_premia_stable_loss(record):
    """
    PREMIA stable loss:
        score = mean_logit_score(chosen) - mean_logit_score(rejected)
    """
    return record["logit_score_chosen"].mean(-1) - record["logit_score_rejected"].mean(-1)

def compute_feature_vector_for_shadow(
    shadow_obj: Dict[str, Any],
    *,
    D: int,
    feature: str = "min_kpp",  # e.g., min_kpp/raw_logprob/.../dpo/ipo/premia/kto/orpo
    stream: str = "chosen",  # "chosen", "rejected", or "both"
    truncation_strategy: str = "tail",
    reference: Optional[Dict[int, List[torch.Tensor]]] = None,
    sample_ids: Optional[Sequence[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    For a single shadow, compute fixed-length vectors (worst-k% of tokens) per sample.

    Returns
    -------
    vectors : Dict[sample_id -> torch.Tensor[feature_len]]
        feature_len is usually D, except for truncation_strategy="average":
        - stream in {"chosen","rejected"} -> feature_len = 1
        - stream == "both"               -> feature_len = 2
        Grouped reducers average_<X>/min_<X>/max_<X> return length D.

        (Implementation note: build_fixed_length_vector() pads short sequences instead
        of returning None, so samples are typically not omitted for being < D.)
        For fixed-size features (e.g., dpo, ipo, premia, kto, orpo), D/truncation_strategy
        are ignored and the raw feature vector is used.
    """

    lr = shadow_obj["loss_records"]
    if reference is not None:
        reference = reference["loss_records"]

        for k, v in reference.items():
            assert len(v) == len(
                lr[k]
            ), f"Reference and shadow loss_records length mismatch for key {k}: {len(v)} vs {len(lr[k])}"

        # Merge reference statistics into shadow loss records
        def merge_reference_stats(
            key: str, shadow: torch.Tensor, reference: torch.Tensor
        ) -> torch.Tensor:
            if key == "index":
                return shadow
            if shadow.dtype == torch.bool:
                return shadow ^ reference
            return shadow - reference

        lr = {
            k: [merge_reference_stats(k, a, b) for a, b in zip(v, reference[k])]
            for k, v in shadow_obj["loss_records"].items()
        }

    vectors: Dict[int, torch.Tensor] = {}
    selected_sample_ids = None if sample_ids is None else {int(x) for x in sample_ids}

    for record in _iter_loss_records(lr):
        record_index = int(record["index"])
        if selected_sample_ids is not None and record_index not in selected_sample_ids:
            continue
        skip_padding = False
        # Per-token features produce 1-D token sequences.
        # - stream in {"chosen","rejected"}: torch.Tensor [T]
        # - stream == "both": tuple(torch.Tensor [Tc], torch.Tensor [Tr])
        if feature == "min_kpp":
            scores = merge_streams(compute_min_kpp_token_scores, record, stream)
        elif feature == "raw_logprob":
            # When stream == "both", _iter_loss_records() has already created
            # record["logprob_both"] = (logprob_chosen_1d, logprob_rejected_1d).
            # For stream in {"chosen","rejected"}, this is just a 1-D tensor.
            scores = record[f"logprob_{stream}"]
        elif feature == "hinge":
            scores = record[f"hinge_{stream}"]
        elif feature == "stable":
            scores = record[f"logit_score_{stream}"]
        elif feature == "logit":
            scores = record[f"top_logit_{stream}"]
        elif feature == "dpo":
            assert stream == "both", "DPO loss requires both streams."
            scores = compute_dpo_loss(record)
            skip_padding = True
        elif feature == "premia":
            assert stream == "both", "Premia loss requires both streams."
            scores = compute_premia_loss(record)
            skip_padding = True
        elif feature == "premia_stable":
            assert stream == "both", "Premia stable loss requires both streams."
            scores = compute_premia_stable_loss(record)
            skip_padding = True
        elif feature == "ipo":
            assert stream == "both", "IPO loss requires both streams."
            scores = compute_ipo_loss(record)
            skip_padding = True
        elif feature == "kto":
            assert stream == "both", "KTO loss requires both streams."
            scores = compute_kto_loss(record)
            skip_padding = True
        elif feature == "orpo":
            assert stream == "both", "ORPO loss requires both streams."
            scores = compute_orpo_loss(record)
            skip_padding = True
        elif feature == "dpo_logprob":
            assert stream == "both", "DPO log-prob requires both streams."
            scores = compute_dpo_logprob(record)
            skip_padding = True
        elif feature == "ipo_logprob":
            assert stream == "both", "IPO log-prob requires both streams."
            scores = compute_ipo_logprob(record)
            skip_padding = True
        elif feature == "kto_logprob":
            assert stream == "both", "KTO log-prob requires both streams."
            scores = compute_kto_logprob(record)
            skip_padding = True
        elif feature == "orpo_logprob":
            assert stream == "both", "ORPO log-prob requires both streams."
            scores = compute_orpo_logprob(record)
            skip_padding = True
        else:
            raise ValueError(f"Unknown feature: {feature}")
        
        # # Normalize direction so larger values correspond to stronger member evidence.
        # # For rejected-only mode we negate directly; for both+half_pad_concat we also
        # # negate the rejected branch before concatenation.
        # #
        # # Note: for stream="both" with truncation_strategy="tail" (or "average"),
        # # the rejected branch is not negated here; it is concatenated/aggregated
        # # as-is by build_fixed_length_vector().
        # if feature in ["raw_logprob", "hinge", "stable", "logit", "min_kpp"]:
        #     if stream == "rejected":
        #         scores = -scores
        #     elif (
        #         stream == "both"
        #         and truncation_strategy == "half_pad_concat"
        #         and isinstance(scores, tuple)
        #         and len(scores) == 2
        #     ):
        #         scores = (scores[0], -scores[1])

        if skip_padding:
            # Sequence-level features: produce a fixed small vector (typically scalar -> [1]).
            vec = torch.as_tensor(scores).reshape(-1)
        else:
            # Fixed-length vector using the selected truncation strategy.
            vec = build_fixed_length_vector(
                scores, D=D, truncation_strategy=truncation_strategy
            )
        if vec is not None:
            vectors[record_index] = vec  # torch.Tensor[feature_len]
    return vectors


def build_feature_cube(
    shadows: Dict[int, Any],
    *,
    D: int,
    feature: str = "min_kpp",
    stream: str = "chosen",
    truncation_strategy: str = "tail",
    reference: Optional[Dict[int, torch.Tensor]] = None,
    n_jobs: int = 1,
    backend: str = "threading",
    dtype: torch.dtype = torch.float64,
    sample_ids: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build V, valid, M across all shadows.

    Parameters
    ----------
    shadows : Dict[int, obj]  # from utils.load_shadow_pickles
    D       : int             # fixed vector length
    feature : str             # any supported feature name
    stream  : str             # "chosen", "rejected", or "both"
    truncation_strategy : str  # "tail", "average", "half_pad_concat", or grouped reducers

    Returns
    -------
    V     : torch.FloatTensor [S, N, feature_len]
        Feature cube. This implementation initializes V to zeros and only writes
        entries for which a feature vector was produced; use `valid` to mask.
    valid : torch.BoolTensor  [S, N]     (True if V[s, n] is valid)
    M     : torch.BoolTensor  [S, N]     (membership matrix from membership_mask)

    Notes
    -----
    - S = number of shadows
    - N = number of samples (taken from membership_mask length; must match across shadows)
    - feature_len may differ from D (e.g. truncation_strategy="average" yields 1 or 2 dims).
    - For samples not present in a shadow, valid=False and V is left at its default value (0.0).
    """
    if not shadows:
        raise ValueError("Empty shadows dict.")

    # Determine S and N
    shadow_ids = sorted(shadows.keys())
    S = len(shadow_ids)

    # Validate membership lengths & pick N
    N_list = []
    for sid in shadow_ids:
        mm = shadows[sid]["membership_mask"]
        if isinstance(mm, torch.Tensor):
            N_list.append(int(mm.numel()))
        elif isinstance(mm, (list, tuple)):
            N_list.append(len(mm))
        else:
            raise TypeError(
                f"Unsupported membership_mask type in shadow {sid}: {type(mm)}"
            )
    if len(set(N_list)) != 1:
        raise ValueError(f"membership_mask lengths differ across shadows: {N_list}")
    N = N_list[0]

    valid = torch.zeros((S, N), dtype=torch.bool)
    M = torch.zeros((S, N), dtype=torch.bool)

    # Fill M
    for s_idx, sid in enumerate(shadow_ids):
        mm = shadows[sid]["membership_mask"]
        mm_t = torch.as_tensor(mm, dtype=torch.bool).reshape(-1)
        if mm_t.numel() != N:
            raise ValueError(f"membership_mask length mismatch in shadow {sid}")
        M[s_idx] = mm_t

    def _compute_vecs_for_shadow(sid: int):
        shadow_obj = shadows[sid]
        vecs = compute_feature_vector_for_shadow(
            shadow_obj,
            D=D,
            feature=feature,
            stream=stream,
            truncation_strategy=truncation_strategy,
            reference=reference,
            sample_ids=sample_ids,
        )
        return sid, vecs

    if n_jobs is None:
        n_jobs = 1
    n_jobs = int(n_jobs)
    sid_to_idx = {sid: idx for idx, sid in enumerate(shadow_ids)}
    requested_jobs = 1 if n_jobs is None else int(n_jobs)
    effective_jobs = 1 if requested_jobs == 0 else (min(max(1, requested_jobs), S) if requested_jobs > 0 else S)
    batch_size = 1 if effective_jobs <= 1 else effective_jobs

    V = None
    feature_len = None

    def _ensure_cube_initialized(batch_results: List[Tuple[int, Dict[int, torch.Tensor]]]) -> None:
        nonlocal V, feature_len
        if V is not None:
            return
        for _, vecs in batch_results:
            if vecs:
                feature_len = next(iter(vecs.values())).size(0)
                break
        if feature_len is None:
            feature_len = D
        V = torch.zeros((S, N, feature_len), dtype=dtype)

    if effective_jobs <= 1:
        for sid in tqdm(shadow_ids, total=S, desc="Building feature cube"):
            batch_results = [_compute_vecs_for_shadow(sid)]
            _ensure_cube_initialized(batch_results)
            for sid_loaded, vecs in batch_results:
                s_idx = sid_to_idx[sid_loaded]
                for sample_id, vec in vecs.items():
                    V[s_idx, sample_id, :] = vec.to(dtype=dtype, device="cpu")
                    valid[s_idx, sample_id] = True
    else:
        for start in tqdm(range(0, S, batch_size), total=(S + batch_size - 1) // batch_size, desc="Building feature cube"):
            batch_shadow_ids = shadow_ids[start : start + batch_size]
            batch_results = Parallel(
                n_jobs=min(effective_jobs, len(batch_shadow_ids)),
                backend=backend,
            )(delayed(_compute_vecs_for_shadow)(sid) for sid in batch_shadow_ids)
            _ensure_cube_initialized(batch_results)
            for sid_loaded, vecs in batch_results:
                s_idx = sid_to_idx[sid_loaded]
                for sample_id, vec in vecs.items():
                    V[s_idx, sample_id, :] = vec.to(dtype=dtype, device="cpu")
                    valid[s_idx, sample_id] = True

    if V is None:
        V = torch.zeros((S, N, D), dtype=dtype)

    return V, valid, M
