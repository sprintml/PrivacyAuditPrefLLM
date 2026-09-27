import os
import sys
import torch
import json
import re
from datetime import datetime
import time
import argparse
from tqdm.auto import tqdm
from tqdm import tqdm
from pathlib import Path

try:
    from threadpoolctl import threadpool_limits, threadpool_info
except Exception:
    threadpool_limits = None
    threadpool_info = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_shadow_pickles, load_reference_pickle
from src.mia_utils import decide_fixed_D, build_feature_cube
from src.lira_utils import (
    compute_lira_for_target,
    compute_rmia_for_target,
    compute_roc_auc,
    tpr_at_fixed_fpr,
)
from src.selected_attack_specs import normalize_attack_setting

_NATIVE_THREADPOOL_LIMITER = None


def _rewrite_deprecated_lira_estimator(setting: dict) -> dict:
    out = dict(setting)
    estimator = str(out.get("lira_covariance_estimator", "")).strip().lower()
    if estimator == "oas":
        out["lira_covariance_estimator"] = "univariate"
        print(
            "[!] lira_covariance_estimator='oas' is disabled due to severe runtime "
            "cost and NaN sensitivity; falling back to 'univariate'."
        )
    return out


def _parse_grouped_truncation_strategy(name: str):
    """
    Parse grouped token reducers of the form:
      average_<X>, min_<X>, max_<X>  (X is a positive integer)
    """
    if name is None:
        return None
    m = re.fullmatch(r"(average|min|max)_(\d+)", str(name).strip().lower())
    if m is None:
        return None
    group_size = int(m.group(2))
    if group_size <= 0:
        raise ValueError(f"Invalid grouped truncation_strategy '{name}': X must be > 0.")
    return m.group(1), group_size


def infer_global_seq_cap(shadows, key: str = "max_seq_len", fallback: int = 256) -> int:
    caps = []
    for sid, obj in shadows.items():
        a = obj.get("args", {})
        if isinstance(a, dict) and key in a and a[key] is not None:
            try:
                caps.append(int(a[key]))
            except Exception:
                pass
    if caps:
        return int(min(caps))
    # soft fallback; we log a note
    print(f"[!] '{key}' not found in shadow args; falling back to {fallback}")
    return int(fallback)

def _load_settings_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Settings file not found: {path}")
    if path.endswith(".jsonl"):
        settings = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                settings.append(json.loads(line))
    else:
        with open(path, "r") as f:
            settings = json.load(f)
        if isinstance(settings, dict):
            settings = [settings]
    if not isinstance(settings, list) or not settings:
        raise ValueError(f"Settings file must contain a non-empty list: {path}")
    return settings


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n", "none", "nan", ""}:
            return False
    return None


def _coerce_setting_types(setting: dict) -> dict:
    out = dict(setting)
    int_keys = {"fixed_d", "torch_threads", "torch_interop_threads"}
    int_keys.update({"num_in_models", "num_out_models"})
    float_keys = {"k_pct", "rmia_log_gamma"}
    bool_keys = {
        "lira_shared_covariance",
        "rmia_offline",
    }

    for k in int_keys:
        if k in out and isinstance(out[k], str):
            v = out[k].strip()
            if v.lower() in {"none", "nan", ""}:
                out[k] = None
            else:
                out[k] = int(v)
    for k in float_keys:
        if k in out and isinstance(out[k], str):
            v = out[k].strip()
            if v.lower() in {"none", "nan", ""}:
                out[k] = None
            else:
                out[k] = float(v)
    for k in bool_keys:
        if k in out:
            parsed = _parse_bool(out[k])
            if parsed is not None:
                out[k] = parsed
    out = _rewrite_deprecated_lira_estimator(out)
    return normalize_attack_setting(out)


def _configure_torch_runtime(parsed) -> tuple[int, int]:
    """
    Configure torch intra-op / inter-op threads once per _score_one run.

    If --torch_threads is not provided, fall back to SLURM_CPUS_PER_TASK when set.
    """
    if parsed.torch_threads is not None:
        n_threads = int(parsed.torch_threads)
    else:
        n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads()))

    if parsed.torch_interop_threads is not None:
        n_interop = int(parsed.torch_interop_threads)
    else:
        n_interop = 1

    n_threads = max(1, n_threads)
    n_interop = max(1, n_interop)

    torch.set_num_threads(n_threads)
    # set_num_interop_threads can only be set early in process lifetime.
    try:
        torch.set_num_interop_threads(n_interop)
    except RuntimeError:
        pass

    # Also cap native threadpools used by NumPy/SciPy/scikit-learn BLAS/OpenMP.
    global _NATIVE_THREADPOOL_LIMITER
    if threadpool_limits is not None:
        _NATIVE_THREADPOOL_LIMITER = threadpool_limits(limits=n_threads)

    return n_threads, n_interop


def _process_target_with_args(
    t,
    V,
    valid,
    M,
    attack_name,
    parsed,
    n_min,
    ridge,
    max_retries,
):
    with torch.inference_mode():
        if attack_name == "LiRA":
            llr_t, scored_t, _ = compute_lira_for_target(
                V,
                valid,
                M,
                target_idx=t,
                n_min=n_min,
                ridge=ridge,
                max_retries=max_retries,
                covariance_estimator=parsed.lira_covariance_estimator,
                shared_covariance=parsed.lira_shared_covariance,
                num_in_models=parsed.num_in_models,
                num_out_models=parsed.num_out_models,
            )
        elif attack_name == "RMIA":
            llr_t, scored_t, _ = compute_rmia_for_target(
                V,
                valid,
                M,
                target_idx=t,
                n_min=n_min,
                mode=parsed.rmia_mode,
                version=parsed.rmia_version,
                log_gamma=parsed.rmia_log_gamma,
                offline=parsed.rmia_offline,
                num_in_models=parsed.num_in_models,
                num_out_models=parsed.num_out_models,
            )
        else:
            raise ValueError(f"Unknown attack name: {attack_name}")

    # Ground-truth label for each sample under target t
    y_t = M[t].to(torch.int32)

    keep = scored_t
    if keep.any():
        sample_idx_t = torch.arange(M.shape[1], device=keep.device, dtype=torch.long)
        return llr_t[keep], y_t[keep], sample_idx_t[keep], int(keep.sum().item())
    else:
        return None, None, None, 0


def _extract_special_canary_metadata(shadows):
    if not shadows:
        return None
    first_shadow = shadows[sorted(shadows.keys())[0]]
    meta = first_shadow.get("special_canaries")
    if not isinstance(meta, dict):
        return None
    indices = meta.get("indices", [])
    try:
        canary_indices = sorted({int(idx) for idx in indices})
    except Exception:
        canary_indices = []
    if not canary_indices:
        return None
    return {
        "base_dataset_size": meta.get("base_dataset_size"),
        "augmented_dataset_size": meta.get("augmented_dataset_size"),
        "counts": dict(meta.get("counts", {})),
        "indices": canary_indices,
        "kinds": list(meta.get("kinds", [])),
        "seed": meta.get("seed"),
    }


def _compute_subset_metrics(y_all, llr_all, sample_idx_all, subset_indices):
    if sample_idx_all is None or subset_indices is None:
        return None
    subset_values = {int(idx) for idx in subset_indices}
    if not subset_values:
        return None
    keep = torch.tensor(
        [int(idx) in subset_values for idx in sample_idx_all.detach().cpu().tolist()],
        dtype=torch.bool,
        device=sample_idx_all.device,
    )
    if not bool(keep.any()):
        return None
    llr_subset = llr_all[keep]
    y_subset = y_all[keep]
    if llr_subset.numel() == 0:
        return None
    pos = int((y_subset == 1).sum().item())
    neg = int((y_subset == 0).sum().item())
    if pos == 0 or neg == 0:
        return {
            "total_pairs_scored": int(llr_subset.numel()),
            "num_positive": pos,
            "num_negative": neg,
        }
    fpr, tpr, auroc = compute_roc_auc(y_subset, llr_subset)
    if auroc < 0.5:
        fpr, tpr, auroc = compute_roc_auc(y_subset, -llr_subset)
    tpr_dict = tpr_at_fixed_fpr(fpr, tpr, targets=[1e-2, 1e-3, 1e-4])
    return {
        "total_pairs_scored": int(llr_subset.numel()),
        "num_positive": pos,
        "num_negative": neg,
        "auroc": float(auroc),
        "tpr@fpr=1e-2": float(tpr_dict.get("0.01", float("nan"))),
        "tpr@fpr=1e-3": float(tpr_dict.get("0.001", float("nan"))),
        "tpr@fpr=1e-4": float(tpr_dict.get("0.0001", float("nan"))),
    }


def _score_one(
    parsed,
    shadows,
    reference,
    *,
    utility_dict,
    model_info,
    extractable_info,
    L_cap,
    feature_cache,
):
    normalized = _rewrite_deprecated_lira_estimator(vars(parsed))
    parsed = argparse.Namespace(**normalize_attack_setting(normalized))
    print("\n[*] Loaded Configuration:")
    for k, v in vars(parsed).items():
        print(f"{k:30}: {v}")
    print("")

    # Run one configuration at a time; use PyTorch threading inside each run.
    n_threads, n_interop = _configure_torch_runtime(parsed)
    print(
        f"[*] Torch runtime threads: intra_op={n_threads}, inter_op={n_interop} "
        "(per _score_one run)"
    )
    if threadpool_info is not None:
        native_infos = threadpool_info()
        if native_infos:
            capped = ", ".join(
                f"{info.get('internal_api', 'native')}={info.get('num_threads', '?')}"
                for info in native_infos
            )
            print(f"[*] Native threadpools after cap: {capped}")

    # 0) Sanity checks
    fixed_size_features = {"dpo", "ipo", "premia", "premia_stable", "kto", "orpo"}
    grouped_trunc = _parse_grouped_truncation_strategy(parsed.truncation_strategy)
    valid_base_trunc = {"tail", "average", "half_pad_concat"}
    if parsed.truncation_strategy not in valid_base_trunc and grouped_trunc is None:
        raise ValueError(
            "Unknown truncation_strategy: "
            f"{parsed.truncation_strategy}. "
            "Supported values: tail, average, half_pad_concat, "
            "average_<X>, min_<X>, max_<X>."
        )
    is_average_style = parsed.truncation_strategy == "average" or (
        grouped_trunc is not None and grouped_trunc[0] == "average"
    )
    assert not (
        parsed.feature in fixed_size_features and is_average_style
    ), (
        f"{parsed.feature.upper()} feature does not support 'average' style truncation strategy."
    )
    if parsed.attack_method.lower() == "rmia" and parsed.rmia_mode == "multivariate_pair":
        assert parsed.stream == "both", "rmia_mode=multivariate_pair requires --stream both."
        assert (
            parsed.truncation_strategy == "half_pad_concat"
        ), "rmia_mode=multivariate_pair requires --truncation_strategy half_pad_concat."
    if (
        parsed.attack_method.lower() == "lira"
        and parsed.lira_covariance_estimator == "pair_diff"
    ):
        assert parsed.stream == "both", "lira_covariance_estimator=pair_diff requires --stream both."
        assert (
            parsed.truncation_strategy in {"average", "half_pad_concat"}
            or grouped_trunc is not None
        ), (
            "lira_covariance_estimator=pair_diff requires --truncation_strategy "
            "in {average, half_pad_concat, average_<X>, min_<X>, max_<X>} "
            "so chosen/rejected halves are well-defined."
        )

    # 2) Infer global seq cap, decide D
    D = decide_fixed_D(max_seq_len=L_cap, k_pct=parsed.k_pct, fixed_d=parsed.fixed_d)
    print(f"[*] Global seq cap (min max_seq_len across shadows): {L_cap}")
    print(
        f"[*] Fixed feature length D: {D} (feature='{parsed.feature}', stream='{parsed.stream}')\n"
    )

    cache_key = (
        parsed.feature,
        parsed.stream,
        parsed.truncation_strategy,
        int(D),
        "ref" if reference is not None else "noref",
    )
    if cache_key in feature_cache:
        V, valid, M = feature_cache[cache_key]
    else:
        # 3) Build feature cube (V), validity (valid), and membership (M)
        V, valid, M = build_feature_cube(
            shadows,
            D=D,
            feature=parsed.feature,
            stream=parsed.stream,
            truncation_strategy=parsed.truncation_strategy,
            reference=reference,
            # Keep this sequential to avoid nested parallel oversubscription.
            n_jobs=1,
            backend="threading",
        )
        feature_cache[cache_key] = (V, valid, M)

    S, N, _ = V.shape
    total_pairs = S * N
    num_valid = int(valid.sum().item())
    pct_valid = 100.0 * num_valid / max(1, total_pairs)

    print("[*] Feature cube summary:")
    print(f"    V shape               : {tuple(V.shape)}  (S, N, D)")
    print(f"    valid shape           : {tuple(valid.shape)}")
    print(f"    M (membership) shape  : {tuple(M.shape)}")
    print(f"    Valid (shadow,sample) : {num_valid} / {total_pairs} ({pct_valid:.2f}%)")

    # Per-shadow validity stats
    per_shadow_valid = valid.sum(dim=1)  # [S]
    min_valid = int(per_shadow_valid.min().item())
    max_valid = int(per_shadow_valid.max().item())
    mean_valid = float(per_shadow_valid.float().mean().item())
    print(
        f"    Per-shadow valid count: min={min_valid}, max={max_valid}, mean={mean_valid:.1f}"
    )

    # Basic membership balance check
    members_per_shadow = M.sum(dim=1)  # [S]
    nonmembers_per_shadow = N - members_per_shadow
    print(
        f"    Members per shadow    : min={int(members_per_shadow.min())}, "
        f"max={int(members_per_shadow.max())}, "
        f"mean={float(members_per_shadow.float().mean()):.1f}"
    )
    print(
        f"    Non-members per shadow: min={int(nonmembers_per_shadow.min())}, "
        f"max={int(nonmembers_per_shadow.max())}, "
        f"mean={float(nonmembers_per_shadow.float().mean()):.1f}"
    )

    n_min = 1
    ridge = 1e-6
    max_retries = 3
    attack_name = {
        "lira": "LiRA",
        "rmia": "RMIA",
    }[parsed.attack_method.lower()]

    print(f"[*] Running rotating-target {attack_name} (leave-one-out, sequential targets):")
    t0 = time.time()
    results = [
        _process_target_with_args(t, V, valid, M, attack_name, parsed, n_min, ridge, max_retries)
        for t in tqdm(range(S), desc=f"{attack_name} targets", unit="target")
    ]
    print(f"[*] Attacks evaluated: {len(results)} / {S}")
    print(f"[*] Target scoring done in {time.time() - t0:.2f}s")

    # Collect results
    all_llr, all_lbl, all_sample_idx, per_target_scored = [], [], [], []

    for llr_t, y_t, sample_idx_t, count in results:
        if llr_t is not None:
            all_llr.append(llr_t)
            all_lbl.append(y_t)
            all_sample_idx.append(sample_idx_t)
        per_target_scored.append(count)

    # Concatenate global pairs
    if all_llr:
        llr_all = torch.cat(all_llr, dim=0)
        y_all = torch.cat(all_lbl, dim=0)
        sample_idx_all = torch.cat(all_sample_idx, dim=0) if all_sample_idx else None
        total_pairs = int(llr_all.numel())
        pos = int((y_all == 1).sum().item())
        neg = total_pairs - pos
        print(f"\n[*] {attack_name} scoring complete:")
        print(f"    Scored pairs (global): {total_pairs}  (pos={pos}, neg={neg})")
        print(
            f"    Per-target scored     : min={min(per_target_scored)}, "
            f"max={max(per_target_scored)}, mean={sum(per_target_scored)/len(per_target_scored):.1f}"
        )
    else:
        print(
            "\n[!] No (target, sample) pairs met the criteria for scoring. "
            "Consider lowering D (via --k_pct or --fixed_d) or n_min."
        )
        return

    t0 = time.time()
    fpr, tpr, auroc = compute_roc_auc(y_all, llr_all)
    # Some feature/attack combinations can naturally invert score direction;
    # flip once so reported AUROC is always >= 0.5.
    if auroc < 0.5:
        fpr, tpr, auroc = compute_roc_auc(y_all, -llr_all)
    tpr_dict = tpr_at_fixed_fpr(fpr, tpr, targets=[1e-2, 1e-3, 1e-4])
    print(f"[*] ROC/AUC computed in {time.time() - t0:.2f}s")

    print(f"    AUROC                 : {auroc:.4f}")
    print(f"    TPR@FPR=1e-2         : {tpr_dict.get('0.01', float('nan')):.4f}")
    print(f"    TPR@FPR=1e-3         : {tpr_dict.get('0.001', float('nan')):.4f}")
    print(f"    TPR@FPR=1e-4         : {tpr_dict.get('0.0001', float('nan')):.4f}")

    special_canary_meta = _extract_special_canary_metadata(shadows)
    canary_metrics = None
    if special_canary_meta is not None and sample_idx_all is not None:
        canary_metrics = _compute_subset_metrics(
            y_all=y_all,
            llr_all=llr_all,
            sample_idx_all=sample_idx_all,
            subset_indices=special_canary_meta.get("indices"),
        )
        if canary_metrics is not None:
            print("[*] Canary-only subset metrics:")
            print(f"    Canary pairs scored   : {int(canary_metrics.get('total_pairs_scored', 0))}")
            if "auroc" in canary_metrics:
                print(f"    Canary AUROC          : {float(canary_metrics['auroc']):.4f}")
                print(f"    Canary TPR@1e-2       : {float(canary_metrics.get('tpr@fpr=1e-2', float('nan'))):.4f}")

    # Filenames
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    count_tag = ""
    if parsed.num_in_models is not None or parsed.num_out_models is not None:
        in_count = 1 if parsed.num_in_models is None else int(parsed.num_in_models)
        out_count = 1 if parsed.num_out_models is None else int(parsed.num_out_models)
        count_tag = f"_IN{in_count}_OUT{out_count}"
    tag = f"{parsed.feature}_{parsed.stream}_{attack_name}_D{D}{count_tag}"
    out_dir = parsed.path_to_pickle.rstrip("/")

    pkl_path = os.path.join(out_dir, f"new_results_{tag}_{ts}.pkl")

    # Save PKL results
    results = {
        "args": vars(parsed),
        "num_shadows": S,
        "num_samples_per_shadow": N,
        "fixed_D": D,
        "total_pairs_scored": total_pairs,
        "num_positive": pos,
        "num_negative": neg,
        "y_all": y_all,
        "llr_all": llr_all,
        "utility": utility_dict,
        "model_info": model_info,
        "extractable_info": extractable_info,
        "special_canaries": special_canary_meta,
        "canary_metrics": canary_metrics,
    }
    t0 = time.time()
    with open(pkl_path, "wb") as f:
        torch.save(results, f)
    print(f"[*] Results saved in {time.time() - t0:.2f}s")
    print(f"[*] Saved results to: {pkl_path}")


def main(args=None):
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path_to_pickle",
        type=str,
        required=True,
        help="Directory containing shadow_*.pkl files",
    )

    ap.add_argument(
        "--reference_model_path",
        type=str,
        default=None,
        help="Path to reference model (if applicable)",
    )
    ap.add_argument(
        "--settings_file",
        type=str,
        default=None,
        help="Optional JSON/JSONL settings file to run multiple configs with one load.",
    )

    # Feature selection
    ap.add_argument(
        "--feature",
        type=str,
        default="raw_logprob",
        choices=[
            "min_kpp",
            "raw_logprob",
            "hinge",
            "stable",
            "logit",
            "dpo",
            "ipo",
            "premia",
            "premia_stable",
            "kto",
            "orpo",
            "dpo_logprob",
            "ipo_logprob",
            "kto_logprob",
            "orpo_logprob",
        ],
        help="Scoring feature",
    )
    ap.add_argument(
        "--stream",
        type=str,
        default="chosen",
        choices=["chosen", "rejected", "both"],
        help="Which response stream to use for feature computation",
    )

    # Fixed-length vector settings
    ap.add_argument(
        "--k_pct",
        type=float,
        default=10.0,
        help="Percent of worst tokens to keep (used to derive D if --fixed_d is not set)",
    )
    ap.add_argument(
        "--fixed_d",
        type=int,
        default=None,
        help="Optional fixed D override; if set, ignores --k_pct-based D.",
    )
    ap.add_argument(
        "--truncation_strategy",
        type=str,
        default="tail",
        help=(
            "Token-to-vector strategy: tail, average, half_pad_concat, "
            "or grouped reducers average_<X>/min_<X>/max_<X> "
            "(group contiguous tokens in blocks of X, reduce each block, then pad/tail)."
        ),
    )

    ap.add_argument(
        "--attack_method",
        type=str,
        default="lira",
        choices=["lira", "rmia"],
        help="Attack method to use: 'lira' or 'rmia' (default: 'lira')",
    )

    # LiRA
    ap.add_argument(
        "--lira_covariance_estimator",
        type=str,
        default="univariate",
        choices=[
            "empirical",
            "diagonal",
            "univariate",
            "univariate_fixed",
            "univariate_alt",
            "pair_diff",
        ],
        help=(
            "Covariance estimator to use (default: 'univariate'). "
            "'pair_diff' runs LiRA separately on chosen/rejected halves and subtracts."
        ),
    )
    ap.add_argument(
        "--lira_shared_covariance",
        action="store_true",
        default=False,
        help="Use shared covariance for IN/OUT (default: False)",
    )

    # RMIA
    ap.add_argument(
        "--rmia_mode",
        type=str,
        default="univariate",
        choices=["univariate", "multivariate", "multivariate_exact", "multivariate_pair"],
        help=(
            "RMIA mode to use: 'univariate', 'multivariate' (legacy), "
            "'multivariate_exact' (paper-faithful multivariate formulation), "
            "or 'multivariate_pair' (exact multivariate run separately on chosen/rejected halves; "
            "requires --stream both and --truncation_strategy half_pad_concat)."
        ),
    )
    ap.add_argument(
        "--rmia_version",
        type=str,
        default="standard",
        choices=["standard", "info"],
        help="RMIA version to use (default: 'standard')",
    )
    ap.add_argument(
        "--rmia_log_gamma",
        type=float,
        default=0.1,
        help="RMIA log_gamma parameter (default: 0.1). Must be > 0. Only for 'standard' version.",
    )

    ap.add_argument(
        "--rmia_offline",
        action="store_true",
        default=False,
            help="RMIA offline mode (default: False)"
    )
    # Performance / threading
    ap.add_argument(
        "--torch_threads",
        type=int,
        default=None,
        help="torch.set_num_threads value (per-process).",
    )
    ap.add_argument(
        "--torch_interop_threads",
        type=int,
        default=None,
        help="torch.set_num_interop_threads value (per-process).",
    )
    ap.add_argument(
        "--num_in_models",
        type=int,
        default=None,
        help="Optional number of IN shadow models to use per scored sample. Defaults to the legacy single-IN behavior.",
    )
    ap.add_argument(
        "--num_out_models",
        type=int,
        default=None,
        help="Optional number of OUT shadow models to use per scored sample. Defaults to the legacy single-OUT behavior.",
    )

    parsed_dict = _rewrite_deprecated_lira_estimator(vars(ap.parse_args(args=args)))
    parsed = argparse.Namespace(**normalize_attack_setting(parsed_dict))

    if parsed.settings_file:
        settings = _load_settings_file(parsed.settings_file)
    else:
        settings = None

    # Validate light-weight inputs before the expensive shadow load so jobs fail
    # fast when the settings file or reference path is missing.
    if parsed.reference_model_path and not os.path.exists(parsed.reference_model_path):
        raise FileNotFoundError(f"Reference model path not found: {parsed.reference_model_path}")

    # 1) Load shadows
    shadows = load_shadow_pickles(parsed.path_to_pickle)
    shadow_ids = sorted(shadows.keys())
    S = len(shadow_ids)
    print(f"[*] Loaded {S} shadows from: {parsed.path_to_pickle}")

    utility_dict = {sid: shadows[sid]["utility"] for sid in shadow_ids}
    model_info = shadows[shadow_ids[0]]["args"]
    extractable_info = {
        k: [shadows[sid]["loss_records"][k] for sid in shadow_ids]
        for k in shadows[shadow_ids[0]]["loss_records"].keys()
        if "log_pz_" in k
    }

    if parsed.reference_model_path:
        reference = load_reference_pickle(parsed.reference_model_path)
        print(f"[*] Loaded reference model from: {parsed.reference_model_path}")
    else:
        reference = None
        print(f"[*] No reference model path provided.")

    # 2) Infer global seq cap
    L_cap = infer_global_seq_cap(shadows, key="max_seq_len", fallback=256)
    feature_cache = {}
    if settings is not None:
        base = vars(parsed).copy()
        base.pop("settings_file", None)
        allowed = set(base.keys())
        for idx, setting in enumerate(settings):
            if not isinstance(setting, dict):
                raise ValueError(f"Settings entry {idx} is not a dict.")
            setting = _coerce_setting_types(setting)
            unknown = set(setting.keys()) - allowed
            if unknown:
                raise ValueError(
                    f"Settings entry {idx} contains unknown keys: {sorted(unknown)}"
                )
            merged = base.copy()
            merged.update(setting)
            parsed_one = argparse.Namespace(**normalize_attack_setting(merged))
            _score_one(
                parsed_one,
                shadows,
                reference,
                utility_dict=utility_dict,
                model_info=model_info,
                extractable_info=extractable_info,
                L_cap=L_cap,
                feature_cache=feature_cache,
            )
    else:
        _score_one(
            parsed,
            shadows,
            reference,
            utility_dict=utility_dict,
            model_info=model_info,
            extractable_info=extractable_info,
            L_cap=L_cap,
            feature_cache=feature_cache,
        )
    print(f"[*] Total runtime: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
