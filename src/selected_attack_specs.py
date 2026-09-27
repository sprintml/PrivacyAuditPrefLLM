from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Mapping


# Shared LiRA variant comparison grid for reusable attack settings. Keep these
# as plain lists so new estimators or truncation modes can be added in one
# place. Scorer-only extras such as `pair_diff` can still be evaluated
# separately without changing this canonical grid.
LIRA_RANK_COMPARE_VARIANTS: list[tuple[str, str]] = [
    ("univariate", "tail"),
    ("univariate", "average"),
    ("empirical", "average"),
]
LIRA_RANK_COMPARE_ESTIMATORS: list[str] = [
    "univariate",
    "empirical",
]
LIRA_RANK_COMPARE_TRUNCATION_STRATEGIES: list[str] = [
    "tail",
    "average",
]
LIRA_RANK_COMPARE_FEATURES: list[str] = [
    "stable",
]
SELECTED_ATTACK_SPECS: dict[str, dict[str, Any]] = {
    "LiRA-W": {
        "attack_method": "lira",
        "stream": "chosen",
        "feature": "hinge",
        "lira_covariance_estimator": "univariate_fixed",
        "truncation_strategy": "tail",
    },
    "RMIA-W": {
        "attack_method": "rmia",
        "stream": "chosen",
        "feature": "raw_logprob",
        "rmia_log_gamma": 0.0,
        "rmia_version": "standard",
        "truncation_strategy": "tail",
        "rmia_mode": "univariate",
    },
    "InfoRMIA-W": {
        "attack_method": "rmia",
        "stream": "chosen",
        "feature": "raw_logprob",
        "rmia_version": "info",
        "truncation_strategy": "tail",
        "rmia_mode": "univariate",
    },
    "LiRA-L": {
        "attack_method": "lira",
        "stream": "rejected",
        "feature": "hinge",
        "lira_covariance_estimator": "univariate_fixed",
        "truncation_strategy": "tail",
    },
    "RMIA-L": {
        "attack_method": "rmia",
        "stream": "rejected",
        "feature": "raw_logprob",
        "rmia_log_gamma": 0.0,
        "rmia_version": "standard",
        "truncation_strategy": "tail",
        "rmia_mode": "univariate",
    },
    "InfoRMIA-L": {
        "attack_method": "rmia",
        "stream": "rejected",
        "feature": "raw_logprob",
        "rmia_version": "info",
        "truncation_strategy": "tail",
        "rmia_mode": "univariate",
    },
    "LiRA-DPO": {
        "attack_method": "lira",
        "stream": "both",
        "feature": "dpo",
        "lira_covariance_estimator": "univariate",
        "truncation_strategy": "tail",
    },
    "RMIA-DPO": {
        "attack_method": "rmia",
        "stream": "both",
        "feature": "dpo",
        "rmia_log_gamma": 0.0,
        "rmia_version": "standard",
        "truncation_strategy": "tail",
        "rmia_mode": "multivariate_exact",
    },
    "InfoRMIA-DPO": {
        "attack_method": "rmia",
        "stream": "both",
        "feature": "dpo",
        "rmia_version": "info",
        "truncation_strategy": "tail",
        "rmia_mode": "multivariate_exact",
    },
    "LiRA-PREMIA": {
        "attack_method": "lira",
        "stream": "both",
        "feature": "premia",
        "lira_covariance_estimator": "univariate",
        "truncation_strategy": "tail",
    },
    "RMIA-PREMIA": {
        "attack_method": "rmia",
        "stream": "both",
        "feature": "premia",
        "rmia_log_gamma": 0.0,
        "rmia_version": "standard",
        "truncation_strategy": "tail",
        "rmia_mode": "multivariate_exact",
    },
    "InfoRMIA-PREMIA": {
        "attack_method": "rmia",
        "stream": "both",
        "feature": "premia",
        "rmia_version": "info",
        "truncation_strategy": "tail",
        "rmia_mode": "multivariate_exact",
    },
    "LiRA-J2": {
        "attack_method": "lira",
        "stream": "both",
        "feature": "raw_logprob",
        "lira_covariance_estimator": "pair_diff",
        "truncation_strategy": "half_pad_concat",
    },
    "RMIA-J2": {
        "attack_method": "rmia",
        "stream": "both",
        "feature": "raw_logprob",
        "rmia_log_gamma": 1.0,
        "rmia_version": "standard",
        "truncation_strategy": "half_pad_concat",
        "rmia_mode": "multivariate_pair",
    },
    "InfoRMIA-J2": {
        "attack_method": "rmia",
        "stream": "both",
        "feature": "raw_logprob",
        "rmia_version": "info",
        "truncation_strategy": "half_pad_concat",
        "rmia_mode": "multivariate_pair",
    },
}


FEATURE_FAMILY_FEATURES: list[str] = [
    "stable",
    "hinge",
    "logit",
    "raw_logprob",
    "min_kpp",
    "ipo",
    "dpo",
    "premia",
    "premia_stable",
    "kto",
    "orpo",
]
J2_MULTIVARIATE_FEATURES: list[str] = [
    "stable",
    "hinge",
    "logit",
    "raw_logprob",
    "min_kpp",
]


FEATURE_FAMILY_SPECS: list[tuple[str, dict[str, str]]] = [
    ("LiRA", {"attack_method": "lira", "stream": "both"}),
    ("RMIA", {"attack_method": "rmia", "stream": "both", "rmia_version": "standard"}),
    ("InfoRMIA", {"attack_method": "rmia", "stream": "both", "rmia_version": "info"}),
]


def normalize_feature_name(value: Any) -> str:
    text = str(value).strip().lower()
    alias_map = {
        "ipo_logprob": "ipo",
        "dpo_logprob": "dpo",
        "kto_logprob": "kto",
        "orpo_logprob": "orpo",
    }
    return alias_map.get(text, text)


def _parse_grouped_truncation_strategy(name: Any) -> tuple[str, int] | None:
    if name is None:
        return None
    match = re.fullmatch(r"(average|min|max)_(\d+)", str(name).strip().lower())
    if match is None:
        return None
    group_size = int(match.group(2))
    if group_size <= 0:
        return None
    return match.group(1), group_size


def normalize_attack_setting(setting: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(setting)
    attack_method = str(out.get("attack_method", "")).strip().lower()
    stream = str(out.get("stream", "")).strip().lower()
    truncation_strategy = str(out.get("truncation_strategy", "tail")).strip().lower()
    lira_covariance_estimator = str(out.get("lira_covariance_estimator", "")).strip().lower()
    rmia_mode = str(out.get("rmia_mode", "")).strip().lower()
    grouped_trunc = _parse_grouped_truncation_strategy(truncation_strategy)

    if attack_method:
        out["attack_method"] = attack_method
    if stream:
        out["stream"] = stream
    if truncation_strategy:
        out["truncation_strategy"] = truncation_strategy
    if lira_covariance_estimator:
        out["lira_covariance_estimator"] = lira_covariance_estimator
    if rmia_mode:
        out["rmia_mode"] = rmia_mode

    if attack_method == "lira" and lira_covariance_estimator == "pair_diff":
        out["stream"] = "both"
        if truncation_strategy not in {"average", "half_pad_concat"} and grouped_trunc is None:
            out["truncation_strategy"] = "half_pad_concat"

    if attack_method == "rmia" and rmia_mode == "multivariate_pair":
        out["stream"] = "both"
        out["truncation_strategy"] = "half_pad_concat"

    return out


def _build_attack_setting_rows(attack_specs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw_attack_cfg in attack_specs:
        attack_cfg = normalize_attack_setting(raw_attack_cfg)
        stream = str(attack_cfg.get("stream", "both"))
        lira_covariance_estimator = attack_cfg.get("lira_covariance_estimator", "None")
        truncation_strategy = attack_cfg.get("truncation_strategy", "tail")
        out.append(
            {
                "fixed_d": "2048" if stream == "both" else "1024",
                "feature": attack_cfg.get("feature"),
                "stream": stream,
                "attack_method": attack_cfg.get("attack_method"),
                "rmia_mode": attack_cfg.get("rmia_mode", "None"),
                "rmia_version": attack_cfg.get("rmia_version", "None"),
                "rmia_log_gamma": str(attack_cfg.get("rmia_log_gamma", "nan")),
                "rmia_offline": "False",
                "lira_covariance_estimator": lira_covariance_estimator,
                "lira_shared_covariance": "True",
                "truncation_strategy": truncation_strategy,
            }
        )
    return out


def build_named_selected_attack_settings() -> list[dict[str, Any]]:
    return _build_attack_setting_rows(SELECTED_ATTACK_SPECS.values())


def build_j2_feature_ablation_attack_settings(
    features: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    feature_list = [normalize_feature_name(feature) for feature in (features or J2_MULTIVARIATE_FEATURES)]
    j2_specs = [
        SELECTED_ATTACK_SPECS["LiRA-J2"],
        SELECTED_ATTACK_SPECS["RMIA-J2"],
        SELECTED_ATTACK_SPECS["InfoRMIA-J2"],
    ]
    expanded_specs: list[dict[str, Any]] = []
    for base_spec in j2_specs:
        for feature in feature_list:
            expanded_specs.append({**base_spec, "feature": feature})
    return _build_attack_setting_rows(expanded_specs)


def build_selected_attack_settings() -> list[dict[str, Any]]:
    # Scoring currently keeps only the canonical selected-attack bundle.
    #
    # Paused scorer-only J2 feature sweep:
    # - LiRA-J2-stable / RMIA-J2-stable / InfoRMIA-J2-stable
    # - LiRA-J2-hinge / RMIA-J2-hinge / InfoRMIA-J2-hinge
    # - LiRA-J2-logit / RMIA-J2-logit / InfoRMIA-J2-logit
    # - LiRA-J2-min_kpp / RMIA-J2-min_kpp / InfoRMIA-J2-min_kpp
    seen: set[tuple[tuple[str, str], ...]] = set()
    merged: list[dict[str, Any]] = []
    for setting in [
        *build_named_selected_attack_settings(),
        # *build_j2_feature_ablation_attack_settings(),
    ]:
        key = tuple(sorted((str(k), str(v)) for k, v in setting.items()))
        if key in seen:
            continue
        seen.add(key)
        merged.append(setting)
    return merged
