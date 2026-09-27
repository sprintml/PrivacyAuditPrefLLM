#!/usr/bin/env python
"""Print representative canary examples for preference datasets."""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import textwrap
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASETS = [
    "HuggingFaceH4/ultrafeedback_binarized",
    "trl-lib/hh-rlhf-helpful-base",
    "chatbot_arena_2024",
]

CANARY_TYPES = [
    "normal_canaries",
    "random_strings",
    "mislabeled",
    "random_question",
    "reasonable_question",
    "reasonable_question_random_chosen",
]

CANARY_LABELS = {
    "normal_canaries": "Normal canaries",
    "random_strings": "Random strings",
    "mislabeled": "Mislabeled preference examples",
    "random_question": "Random prompt with real responses",
    "reasonable_question": "Real prompt with random responses",
    "reasonable_question_random_chosen": "Real prompt with random preferred response",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and print canary examples using the same canary families "
            "as scripts/train_shadow.py."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names to load. Defaults to the built-in preference datasets.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=200,
        help="Load at most this many train rows per dataset before sampling sources.",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=1,
        help="Number of canaries to generate per canary type and dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed. The dataset index is added to this seed.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help=(
            "Maximum printed characters per prompt/chosen/rejected field. "
            "Use 0 for no truncation."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON array instead of human-readable text.",
    )
    return parser.parse_args()


def random_string_of_length(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    if length <= 0:
        return ""
    return "".join(rng.choice(alphabet) for _ in range(length))


def message_content(messages: Any, index: int) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        if not messages:
            return ""
        item = messages[index] if index < len(messages) else messages[-1]
        if isinstance(item, dict):
            return str(item.get("content", ""))
        return str(item)
    if isinstance(messages, dict):
        return str(messages.get("content", ""))
    return str(messages)


def sample_text(sample: dict[str, Any]) -> tuple[str, str, str]:
    return (
        message_content(sample["prompt"], 0),
        message_content(sample["chosen"], 1),
        message_content(sample["rejected"], 1),
    )


def generate_canaries(
    base_ds: Any,
    *,
    per_type: int,
    seed: int,
    dataset_name: str,
) -> list[dict[str, Any]]:
    if per_type < 0:
        raise ValueError("--per-type must be non-negative.")
    if len(base_ds) == 0:
        raise ValueError(f"Dataset '{dataset_name}' is empty.")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    def source() -> tuple[int, str, str, str]:
        source_index = rng.randrange(len(base_ds))
        prompt, chosen, rejected = sample_text(base_ds[source_index])
        return source_index, prompt, chosen, rejected

    def add(
        kind: str,
        prompt: str,
        chosen: str,
        rejected: str,
        source_index: int | None,
    ) -> None:
        rows.append(
            {
                "dataset_name": dataset_name,
                "kind": kind,
                "kind_label": CANARY_LABELS[kind],
                "source_index": source_index,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    for _ in range(per_type):
        source_index, prompt, chosen, rejected = source()
        add("normal_canaries", prompt, chosen, rejected, source_index)

    for _ in range(per_type):
        source_index, prompt, chosen, rejected = source()
        add(
            "random_strings",
            random_string_of_length(rng, len(prompt)),
            random_string_of_length(rng, len(chosen)),
            random_string_of_length(rng, len(rejected)),
            source_index,
        )

    for _ in range(per_type):
        source_index, prompt, chosen, rejected = source()
        add("mislabeled", prompt, rejected, chosen, source_index)

    for _ in range(per_type):
        source_index, prompt, chosen, rejected = source()
        add(
            "random_question",
            random_string_of_length(rng, len(prompt)),
            chosen,
            rejected,
            source_index,
        )

    for _ in range(per_type):
        source_index, prompt, _chosen, _rejected = source()
        add(
            "reasonable_question",
            prompt,
            random_string_of_length(rng, len(_chosen)),
            random_string_of_length(rng, len(_rejected)),
            source_index,
        )

    for _ in range(per_type):
        source_index, prompt, chosen, rejected = source()
        add(
            "reasonable_question_random_chosen",
            prompt,
            random_string_of_length(rng, len(chosen)),
            rejected,
            source_index,
        )

    return rows


def clean_for_printing(text: Any, max_chars: int) -> str:
    out = " ".join(str(text).split())
    if max_chars > 0 and len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
    return out


def print_text(rows: list[dict[str, Any]], *, max_chars: int) -> None:
    current_dataset = None
    per_kind_count: dict[tuple[str, str], int] = {}
    for row in rows:
        dataset_name = row["dataset_name"]
        if dataset_name != current_dataset:
            if current_dataset is not None:
                print()
            print("=" * 88)
            print(f"Dataset: {dataset_name}")
            current_dataset = dataset_name

        key = (dataset_name, row["kind"])
        per_kind_count[key] = per_kind_count.get(key, 0) + 1
        source = row["source_index"] if row["source_index"] is not None else "none"
        print()
        print(
            f"[{row['kind']}] {row['kind_label']} #{per_kind_count[key]} "
            f"(source_index={source})"
        )
        for field in ("prompt", "chosen", "rejected"):
            value = clean_for_printing(row[field], max_chars)
            print(f"{field}:")
            print(textwrap.indent(value, "  "))


def main() -> int:
    args = parse_args()
    try:
        from src.dataset_utils import get_dataset
    except ModuleNotFoundError as exc:
        if exc.name == "datasets":
            print(
                "Missing dependency: datasets. Activate the project environment "
                "from environment.yaml before running this script.",
                file=sys.stderr,
            )
            return 2
        raise

    rows: list[dict[str, Any]] = []
    for dataset_index, dataset_name in enumerate(args.datasets):
        train_ds, _eval_ds = get_dataset(dataset_name, subset_size=args.subset_size)
        rows.extend(
            generate_canaries(
                train_ds,
                per_type=args.per_type,
                seed=args.seed + dataset_index,
                dataset_name=dataset_name,
            )
        )

    if args.json:
        json.dump(rows, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        print_text(rows, max_chars=args.max_chars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
