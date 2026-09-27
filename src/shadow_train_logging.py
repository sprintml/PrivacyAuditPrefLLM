from __future__ import annotations

from typing import Any, Iterable, Mapping

from transformers import TrainerCallback


TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
FALSE_STRINGS = {"0", "false", "no", "n", "off"}


def parse_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in TRUE_STRINGS:
        return True
    if normalized in FALSE_STRINGS:
        return False
    raise ValueError(
        f"Expected a boolean-like value, but received {value!r}. "
        f"Use one of: {sorted(TRUE_STRINGS | FALSE_STRINGS)}."
    )


def _is_scalar_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def summarize_epoch_metrics(log_history: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not log_history:
        return []

    by_epoch: dict[float, dict[str, Any]] = {}
    for row in log_history:
        if not isinstance(row, Mapping):
            continue
        epoch_value = row.get("epoch")
        if epoch_value is None:
            continue
        try:
            epoch = float(epoch_value)
        except (TypeError, ValueError):
            continue

        summary = by_epoch.setdefault(epoch, {"epoch": epoch})
        for key, value in row.items():
            if _is_scalar_metric(value):
                summary[key] = value

    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def format_epoch_metric_row(metric_row: Mapping[str, Any]) -> str:
    preferred_keys = [
        "epoch",
        "loss",
        "eval_loss",
        "eval_rewards/accuracies",
        "eval_accuracy",
        "eval_runtime",
        "train_runtime",
        "learning_rate",
    ]
    seen = set()
    parts = []
    for key in preferred_keys:
        if key in metric_row and key not in seen:
            parts.append(f"{key}={metric_row[key]:.6g}")
            seen.add(key)
    for key in sorted(metric_row):
        if key in seen or key == "step":
            continue
        value = metric_row[key]
        if _is_scalar_metric(value):
            parts.append(f"{key}={value:.6g}")
    return ", ".join(parts)


def print_epoch_metric_summary(metric_rows: Iterable[Mapping[str, Any]], *, prefix: str = "[*]") -> None:
    rows = list(metric_rows)
    if not rows:
        print(f"{prefix} No epoch-level trainer metrics were recorded.", flush=True)
        return
    print(f"{prefix} Epoch-level trainer metrics:", flush=True)
    for row in rows:
        print(f"{prefix}   {format_epoch_metric_row(row)}", flush=True)


class EpochMetricsLoggerCallback(TrainerCallback):
    def __init__(self, *, prefix: str = "[*]"):
        self.prefix = prefix

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(
                f"{self.prefix} Epoch evaluation: {format_epoch_metric_row(metrics)}",
                flush=True,
            )
        return control


def format_duration_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = seconds / 60.0
    if seconds < 3600.0:
        return f"{seconds:.2f}s ({minutes:.2f}m)"
    hours = minutes / 60.0
    return f"{seconds:.2f}s ({minutes:.2f}m, {hours:.2f}h)"


def log_phase_duration(phase: str, elapsed_seconds: float, *, prefix: str = "[*]") -> None:
    print(
        f"{prefix} Timing | {phase}: {format_duration_seconds(elapsed_seconds)}",
        flush=True,
    )
