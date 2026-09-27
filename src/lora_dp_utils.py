from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset


@dataclass
class LoraState:
    # Canonical unique tensors (deduplicated by shared storage).
    tensors: Dict[str, torch.Tensor]
    # canonical name -> all aliased parameter names sharing that tensor.
    alias_groups: Dict[str, Tuple[str, ...]]


def partition_dataset_into_k_groups(
    dataset: Dataset, num_groups: int, seed: int
) -> List[Dataset]:
    """Deterministically partition a dataset into K disjoint shuffled groups."""
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if len(dataset) < num_groups:
        raise ValueError(
            f"num_groups={num_groups} cannot exceed dataset size={len(dataset)}."
        )

    rng = np.random.default_rng(seed)
    shuffled = np.arange(len(dataset))
    rng.shuffle(shuffled)
    split_indices = np.array_split(shuffled, num_groups)
    return [dataset.select(idx.tolist()) for idx in split_indices]


def extract_lora_state(
    model, adapter_patterns: Sequence[str] = ("lora_", "vera_")
) -> LoraState:
    """
    Extract adapter parameters from a PEFT model as fp32 tensors on CPU.

    Shared tensors are deduplicated to avoid double-counting in clipping and noise.
    """
    named_params = dict(model.named_parameters())
    grouped: Dict[Tuple[int, int, Tuple[int, ...], Tuple[int, ...], str], List[str]] = {}
    for name, param in named_params.items():
        if not _matches_adapter_name(name, adapter_patterns):
            continue
        sig = (
            int(param.untyped_storage().data_ptr()),
            int(param.storage_offset()),
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
        )
        grouped.setdefault(sig, []).append(name)

    if not grouped:
        patterns_str = ",".join(adapter_patterns)
        raise ValueError(
            f"No adapter parameters found for patterns [{patterns_str}]. "
            "Make sure adapter training is enabled."
        )

    tensors: Dict[str, torch.Tensor] = {}
    alias_groups: Dict[str, Tuple[str, ...]] = {}
    for names in grouped.values():
        aliases = tuple(sorted(names))
        canonical = aliases[0]
        tensors[canonical] = (
            named_params[canonical]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clone()
        )
        alias_groups[canonical] = aliases

    return LoraState(tensors=tensors, alias_groups=alias_groups)


@torch.no_grad()
def load_lora_state(model, lora_state: LoraState) -> None:
    """Load adapter tensors into model parameters, respecting alias groups."""
    name_to_canonical = _name_to_canonical(lora_state)
    loaded = 0
    for name, param in model.named_parameters():
        canonical = name_to_canonical.get(name, None)
        if canonical is not None:
            src = lora_state.tensors[canonical]
            param.copy_(src.to(device=param.device, dtype=param.dtype))
            loaded += 1

    expected = len(name_to_canonical)
    if loaded != expected:
        model_names = {n for n, _ in model.named_parameters()}
        missing = sorted(set(name_to_canonical.keys()) - model_names)
        raise ValueError(
            f"Failed to load all adapter tensors. Loaded {loaded}/{expected}. "
            f"Missing keys: {missing[:8]}"
        )


def project_lora_state_toward_init(
    trained_state: LoraState,
    init_state: LoraState,
    clip_factor: float,
    eps: float = 1e-12,
) -> Tuple[LoraState, float, float]:
    """
    Project adapter update (trained - init) onto L2 ball of radius clip_factor.

    Norm is computed over unique canonical tensors only.
    Returns:
      projected_state, update_norm_before_projection, projection_scale
    """
    if clip_factor <= 0:
        raise ValueError(f"clip_factor must be > 0, got {clip_factor}")

    _validate_compatible_states(trained_state, init_state)
    sq_norm = 0.0
    for key in init_state.tensors.keys():
        diff = trained_state.tensors[key] - init_state.tensors[key]
        sq_norm += float(torch.sum(diff * diff).item())
    update_norm = sq_norm**0.5
    scale = min(1.0, clip_factor / (update_norm + eps))

    projected_tensors: Dict[str, torch.Tensor] = {}
    for key in init_state.tensors.keys():
        projected_tensors[key] = init_state.tensors[key] + (
            trained_state.tensors[key] - init_state.tensors[key]
        ) * scale
    return (
        LoraState(
            tensors=projected_tensors,
            alias_groups={k: tuple(v) for k, v in init_state.alias_groups.items()},
        ),
        update_norm,
        scale,
    )


def aggregate_lora_states(lora_states: Sequence[LoraState]) -> LoraState:
    """Average multiple adapter states element-wise over canonical tensors."""
    if len(lora_states) == 0:
        raise ValueError("lora_states must contain at least one state.")
    reference = lora_states[0]
    for state in lora_states[1:]:
        _validate_compatible_states(state, reference)

    agg_tensors: Dict[str, torch.Tensor] = {}
    denom = float(len(lora_states))
    for key in reference.tensors.keys():
        running = torch.zeros_like(reference.tensors[key], dtype=torch.float32)
        for state in lora_states:
            running += state.tensors[key]
        agg_tensors[key] = running / denom
    return LoraState(
        tensors=agg_tensors,
        alias_groups={k: tuple(v) for k, v in reference.alias_groups.items()},
    )


def gaussian_std_for_mu_dp(mu_dp: float, clip_factor: float, num_groups: int) -> float:
    """
    Calibrate isotropic Gaussian noise std for a simple mu-DP target.

    Assumes:
    - each group contribution is clipped to L2 radius `clip_factor`,
    - aggregation is arithmetic mean across `num_groups`.
    Then L2 sensitivity is upper-bounded by 2*clip_factor/num_groups.
    """
    if mu_dp <= 0:
        raise ValueError(f"mu_dp must be > 0, got {mu_dp}")
    if clip_factor <= 0:
        raise ValueError(f"clip_factor must be > 0, got {clip_factor}")
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")

    sensitivity = (2.0 * clip_factor) / float(num_groups)
    return sensitivity / mu_dp


def add_gaussian_noise_to_lora_state(
    lora_state: LoraState, noise_std: float, seed: int | None = None
) -> LoraState:
    """
    Add i.i.d. Gaussian noise N(0, noise_std^2) to canonical tensors.

    Aliased/shared names reuse their canonical tensor and therefore use the same noise.
    """
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}")

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    noisy_tensors: Dict[str, torch.Tensor] = {}
    for key, value in lora_state.tensors.items():
        if noise_std == 0:
            noisy_tensors[key] = value.clone()
        else:
            noise = torch.randn(
                value.shape, dtype=value.dtype, device="cpu", generator=generator
            )
            noisy_tensors[key] = value + noise * noise_std
    return LoraState(
        tensors=noisy_tensors,
        alias_groups={k: tuple(v) for k, v in lora_state.alias_groups.items()},
    )


def _validate_same_keys(lhs_keys, rhs_keys, label: str) -> None:
    lhs = set(lhs_keys)
    rhs = set(rhs_keys)
    if lhs != rhs:
        only_lhs = sorted(lhs - rhs)
        only_rhs = sorted(rhs - lhs)
        raise ValueError(
            f"Adapter states use different {label}. "
            f"Only in first: {only_lhs[:5]} Only in second: {only_rhs[:5]}"
        )


def _validate_compatible_states(lhs: LoraState, rhs: LoraState) -> None:
    _validate_same_keys(lhs.tensors.keys(), rhs.tensors.keys(), label="canonical keys")
    _validate_same_keys(
        lhs.alias_groups.keys(), rhs.alias_groups.keys(), label="alias-group keys"
    )
    for key in lhs.alias_groups.keys():
        if tuple(lhs.alias_groups[key]) != tuple(rhs.alias_groups[key]):
            raise ValueError(
                f"Alias-group mismatch for key '{key}': "
                f"{lhs.alias_groups[key]} vs {rhs.alias_groups[key]}"
            )


def _name_to_canonical(state: LoraState) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for canonical, aliases in state.alias_groups.items():
        for name in aliases:
            out[name] = canonical
    return out


def _matches_adapter_name(name: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if pattern in name:
            return True
    return False
