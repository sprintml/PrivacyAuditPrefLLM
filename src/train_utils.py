import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset
from sklearn.metrics import roc_auc_score
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
try:
    import torch.distributed as dist
except Exception:
    dist = None


@dataclass
class SamplingScheme:
    """Post-processing g_ϕ for (n,p)-discoverable extraction."""

    name: str
    type: str  # 'top_k', 'top_p', 'temperature'
    k: Optional[int] = None  # for top_k
    p: Optional[float] = None  # for top_p
    T: float = 1.0  # temperature


def compute_avg_logprob(records):
    return np.mean([lp.mean().item() for lp in records["logprob_chosen"]]), np.mean(
        [lp.mean().item() for lp in records["logprob_rejected"]]
    )


def compute_preference_accuracy(records):
    correct_train = 0
    correct_test = 0
    count_train = 0
    count_test = 0
    for i in range(len(records["index"])):
        lp_chosen = records["logprob_chosen"][i]
        lp_rejected = records["logprob_rejected"][i]
        if records["is_member"][i]:
            count_train += 1
            if lp_chosen.mean() > lp_rejected.mean():
                correct_train += 1
        else:
            count_test += 1
            if lp_chosen.mean() > lp_rejected.mean():
                correct_test += 1
    return correct_train / count_train if count_train > 0 else 0.0, (
        correct_test / count_test if count_test > 0 else 0.0
    )


def _get_dist_info() -> Tuple[int, int]:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _merge_records(shards: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if not shards:
        return merged
    keys = shards[0].keys()
    for key in keys:
        if key in ("index", "is_member"):
            tensors = [shard[key] for shard in shards]
            merged[key] = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
        else:
            merged_list = []
            for shard in shards:
                merged_list.extend(shard[key])
            merged[key] = merged_list
    return merged


def get_member_indices(
    num_samples: int, num_shadows: int, shadow_id: int, seed: int
) -> np.ndarray:
    """
    Membership (IN/OUT)

    Deterministic, vectorized splitter.
    Each sample belongs to exactly half of the shadows (num_shadows must be even).
    Returns indices of IN (member) samples for the given shadow_id.
    """
    assert num_shadows % 2 == 0, "num_shadows must be even"

    rng = np.random.default_rng(seed)
    scores = rng.uniform(size=(num_shadows, num_samples))  # (S, N)
    sorted_idxs = np.argsort(scores, axis=0)  # (S, N)
    lower = sorted_idxs[: num_shadows // 2, :]  # (S/2, N)
    shadow_in = np.zeros((num_shadows, num_samples), dtype=bool)

    for s in range(num_shadows):
        # sample `j` is in shadow `s` if `s` is among the lowest half for column `j`
        shadow_in[s, :] = np.any(lower == s, axis=0)

    return np.nonzero(shadow_in[shadow_id])[0]


# Adapted from a public privacy-evaluation utility.
def hinge_score(raw_predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert (
        raw_predictions.dim() >= 2
        and labels.dim() == 1
        and raw_predictions.size(0) == len(labels)
    )
    raw_predictions = raw_predictions.to(dtype=torch.float64)

    target_predictions = raw_predictions[torch.arange(len(labels)), ..., labels]
    raw_predictions[torch.arange(len(labels)), ..., labels] = float("-inf")
    return target_predictions - torch.max(raw_predictions, dim=-1).values


def logit_score(
    raw_predictions: torch.Tensor, labels: torch.Tensor, eps: float = 1e-30
) -> torch.Tensor:
    assert (
        raw_predictions.dim() >= 2
        and labels.dim() == 1
        and raw_predictions.size(0) == len(labels)
    )
    raw_predictions = raw_predictions.to(dtype=torch.float64)

    # Original LiRA implementation first calculates probabilities via numerically stable softmax,
    # and then calculates the logit score from probabilities.
    # However, there is no need to calculate all probabilities, thereby avoiding log(exp(...)) operations
    # and using a potentially more appropriate LogSumExp normalization constant.

    # torch.logsumexp works with -inf, hence this version is more memory-efficient
    target_predictions = raw_predictions[torch.arange(len(labels)), ..., labels]
    raw_predictions[torch.arange(len(labels)), ..., labels] = float("-inf")
    return target_predictions - torch.logsumexp(raw_predictions, dim=-1)


@torch.no_grad()
def compute_metrics_by_sample(
    model,
    sample,
    labels_chosen,
    extraction_schemes: Optional[List[SamplingScheme]] = None,
    np_p_list: Tuple[float, ...] = (0.1, 0.5, 0.9, 0.99, 0.999),
):
    res = {}
    out = model(
        input_ids=sample["input_ids"],
        attention_mask=sample["attention_mask"],
        use_cache=False,
    )
    logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)[0]
    lb = labels_chosen[0, 1:].to(logp.device)
    mask = lb != -100
    logits, logp, lb = out.logits[0, :-1, :][mask], logp[mask], lb[mask]
    p = logp.exp()
    lb = lb[(lb != -100)]
    mu = (logp * p).sum(-1)
    sigma = ((p * torch.square(logp)).sum(-1) - torch.square(mu)).sqrt()
    res["mu"] = mu.cpu()
    res["sigma"] = sigma.cpu()
    losses = -F.nll_loss(logp, lb, reduction="none")
    res["logprob"] = losses.cpu()
    res["top_logit"] = logits[torch.arange(lb.shape[0], device=logits.device), lb].cpu()
    res["hinge"] = hinge_score(logits, lb).cpu()
    res["logit_score"] = logit_score(logits, lb).cpu()

    if extraction_schemes:  # (n, p)-extraction
        for sch in extraction_schemes:
            pz, log_pz = prob_of_target_under_scheme(
                logits, lb, sch, return_tokenwise=False
            )

            pz_val = float(pz)
            log_pz_val = float(log_pz.item())
            res[f"pz_{sch.name}"] = torch.tensor(float(pz_val)).cpu()
            res[f"log_pz_{sch.name}"] = torch.tensor(float(log_pz_val)).cpu()

            # expected draws until first success
            res[f"E_queries_{sch.name}"] = torch.tensor(
                float(math.inf if pz <= 0 else math.ceil(1.0 / float(pz)))
            ).cpu()

            for q in np_p_list:
                n_req = n_required_for_p(float(pz), float(q))
                res[f"n_at_p{q}_{sch.name}"] = torch.tensor(float(n_req)).cpu()
                res[f"n_at_p{q}_ceil_{sch.name}"] = torch.tensor(
                    int(math.ceil(n_req)) if math.isfinite(n_req) else int(1e12)
                ).cpu()
    return res


@torch.no_grad()
def compute_metrics(
    model,
    tokenizer,
    ds: Dataset,
    membership_mask: List[bool],
    max_length: int = 1024,
    extraction_schemes: Optional[List[SamplingScheme]] = None,
    np_p_list: Tuple[float, ...] = (0.1, 0.5, 0.9, 0.99),
    gather_across_ranks: bool = False,
) -> Optional[Dict[str, List[Any]]]:
    device = next(model.parameters()).device
    rank = 0
    world_size = 1
    if gather_across_ranks:
        rank, world_size = _get_dist_info()
    records = {
        "index": [],
        "is_member": [],
        "logprob_chosen": [],
        "logprob_rejected": [],
        "mu_chosen": [],
        "mu_rejected": [],
        "sigma_chosen": [],
        "sigma_rejected": [],
        "top_logit_chosen": [],
        "top_logit_rejected": [],
        "hinge_chosen": [],
        "hinge_rejected": [],
        "logit_score_chosen": [],
        "logit_score_rejected": [],
    }
    if extraction_schemes:
        for sch in extraction_schemes:
            for side in ("chosen", "rejected"):
                records[f"pz_{sch.name}_{side}"] = []
                records[f"log_pz_{sch.name}_{side}"] = []
                records[f"E_queries_{sch.name}_{side}"] = []
                for q in np_p_list:
                    records[f"n_at_p{q}_{sch.name}_{side}"] = []
                    records[f"n_at_p{q}_ceil_{sch.name}_{side}"] = []

    if gather_across_ranks and world_size > 1:
        indices = list(range(rank, len(ds), world_size))
    else:
        indices = range(len(ds))

    iterator = tqdm(
        indices,
        desc=f"Computing Loss Records for {len(ds)} samples",
        total=len(indices),
        disable=rank != 0,
    )
    for i in iterator:
        ex = ds[i]
        prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]

        base_len = tokenizer.apply_chat_template(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"].size(1)

        chosen_tok = tokenizer.apply_chat_template(
            chosen,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_dict=True,
        )

        rejected_tok = tokenizer.apply_chat_template(
            rejected,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_dict=True,
        )

        chosen_tok = {k: v.to(device) for k, v in chosen_tok.items()}
        rejected_tok = {k: v.to(device) for k, v in rejected_tok.items()}
        labels_chosen = chosen_tok["input_ids"].clone()
        labels_rejected = rejected_tok["input_ids"].clone()

        for j in range(min(base_len, labels_chosen.size(1))):
            labels_chosen[0, j] = -100
        for j in range(min(base_len, labels_rejected.size(1))):
            labels_rejected[0, j] = -100

        if base_len >= chosen_tok["input_ids"].size(1) or base_len >= rejected_tok[
            "input_ids"
        ].size(1):
            # Degenerate: no prediction tokens
            continue

        records["index"].append(i)
        records["is_member"].append(membership_mask[i])

        # chosen
        res = compute_metrics_by_sample(
            model,
            chosen_tok,
            labels_chosen,
            extraction_schemes=extraction_schemes,
            np_p_list=np_p_list,
        )
        for k, v in list(res.items()):
            if (
                k.startswith("pz_")
                or k.startswith("log_pz_")
                or k.startswith("E_queries_")
                or k.startswith("n_at_p")
            ):
                records[f"{k}_chosen"].append(v)
            else:
                records[f"{k}_chosen"].append(v)

        # rejected
        res = compute_metrics_by_sample(
            model,
            rejected_tok,
            labels_rejected,
            extraction_schemes=extraction_schemes,
            np_p_list=np_p_list,
        )
        for k, v in list(res.items()):
            if (
                k.startswith("pz_")
                or k.startswith("log_pz_")
                or k.startswith("E_queries_")
                or k.startswith("n_at_p")
            ):
                records[f"{k}_rejected"].append(v)
            else:
                records[f"{k}_rejected"].append(v)

    records["is_member"] = torch.tensor(records["is_member"], dtype=torch.bool)
    records["index"] = torch.tensor(records["index"], dtype=torch.long)

    if gather_across_ranks and world_size > 1 and dist is not None and dist.is_initialized():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, records)
        if rank == 0:
            return _merge_records(gathered)
        return None

    return records


def quick_mia_score(loss_records: Dict[str, List[Any]]) -> Dict[str, float]:
    record = loss_records
    y = [int(a) for a in loss_records["is_member"]]
    score = {}
    score["logprob_chosen"] = [a.mean().item() for a in record["logprob_chosen"]]
    score["logprob_rejected"] = [a.mean().item() for a in record["logprob_rejected"]]
    score["kpp_chosen"] = [
        ((a - b) / (c + 1e-6))
        .topk(int(0.25 * len(a)), largest=False)
        .values.mean(-1)
        .item()
        for a, b, c in zip(
            record["logprob_chosen"], record["mu_chosen"], record["sigma_chosen"]
        )
    ]
    score["kpp_rejected"] = [
        ((a - b) / (c + 1e-6))
        .topk(int(0.25 * len(a)), largest=False)
        .values.mean(-1)
        .item()
        for a, b, c in zip(
            record["logprob_rejected"], record["mu_rejected"], record["sigma_rejected"]
        )
    ]

    for k, v in score.items():
        score[k] = torch.tensor(v).nan_to_num(nan=0.0, posinf=1e9, neginf=1e9).numpy()

    res = {}
    a = roc_auc_score(y, score["logprob_chosen"])
    res["auroc_logprob_chosen"] = max(a, 1 - a)
    a = roc_auc_score(y, score["logprob_rejected"])
    res["auroc_logprob_rejected"] = max(a, 1 - a)
    a = roc_auc_score(y, score["kpp_chosen"])
    res["auroc_kpp_chosen"] = max(a, 1 - a)
    a = roc_auc_score(y, score["kpp_rejected"])
    res["auroc_kpp_rejected"] = max(a, 1 - a)
    return res


def _log_softmax_masked(
    logits: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    logits: [L, V], mask: [L, V] bool or None. True = keep, False = drop to -inf.
    Returns log_softmax over masked logits.
    """
    if mask is not None:
        masked = torch.full_like(logits, float("-inf"))
        masked[mask] = logits[mask]
        logits = masked
    return F.log_softmax(logits, dim=-1)


def _topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    # logits: [L, V] -> bool mask [L, V] with top-k True
    _, idx = logits.topk(k, dim=-1)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def _topp_mask_from_probs(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    probs: [L, V] after temperature scaling softmax (not masked yet)
    Keep minimal prefix where cumulative >= p (nucleus), standard shifting rule.
    """
    # sort probs
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    remove = cum > p
    # ensure at least one token kept
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    keep_sorted = ~remove
    # map back to original indices
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep.scatter_(1, sorted_idx, keep_sorted)
    return keep


def prob_of_target_under_scheme(
    logits: torch.Tensor,
    labels: torch.Tensor,
    scheme: SamplingScheme,
    return_tokenwise: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    logits: [L, V] (already filtered to label positions as you do)
    labels: [L]
    Returns (p_z, per_token_probs or None)
    """
    # 1) temperature
    logits_T = logits / max(scheme.T, 1e-8)

    # 2) build mask per scheme
    if scheme.type == "top_k":
        mask = _topk_mask(logits_T, scheme.k if scheme.k is not None else 1)
        log_probs = _log_softmax_masked(logits_T, mask)
    elif scheme.type == "top_p":
        # for top-p, compute probs first to decide nucleus set per step
        probs_T = F.softmax(logits_T, dim=-1)
        mask = _topp_mask_from_probs(probs_T, scheme.p if scheme.p is not None else 0.9)
        # re-normalize on the nucleus support
        log_probs = _log_softmax_masked(logits_T, mask)
    elif scheme.type == "temperature":
        # plain temperature sampling (no truncation)
        log_probs = F.log_softmax(logits_T, dim=-1)
    else:
        raise ValueError(f"Unknown sampling scheme: {scheme.type}")

    # 3) per-token probs of the true target tokens
    L = labels.shape[0]
    token_logp = log_probs[torch.arange(L, device=logits.device), labels]
    # masked-out => zero prob in log-domain
    token_logp = torch.clamp(token_logp, min=-1e9)
    token_logp = token_logp.to(dtype=torch.float64)
    log_pz = token_logp.sum()
    log_pz_clamped = torch.clamp(log_pz, min=-80.0)
    pz = torch.exp(log_pz_clamped)

    if return_tokenwise:
        return pz, log_pz, token_logp
    return pz, log_pz


def n_required_for_p(pz: float, p: float) -> float:
    """
    https://www.arxiv.org/pdf/2410.19482v3
    Eq. (2): n >= log(1-p) / log(1-p_z)
    Returns float.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return math.inf
    if pz <= 0.0:
        return math.inf
    if pz >= 1.0:
        return 1.0
    log_1_pz = math.log(1.0 - float(pz))
    if log_1_pz >= 0:
        return math.inf
    return math.log(1.0 - p) / log_1_pz
