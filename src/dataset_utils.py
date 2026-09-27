import re
from typing import Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict


def map_to_standard_format(x):
    prompt = re.sub(r"\s+", " ", x["prompt"]).strip().replace("\\", "")
    chosen = re.sub(r"\s+", " ", x["chosen"]).strip().replace("\\", "")
    rejected = re.sub(r"\s+", " ", x["rejected"]).strip().replace("\\", "")
    return {
        "prompt": [{"content": prompt, "role": "user"}],
        "chosen": [
            {"content": prompt, "role": "user"},
            {"content": chosen, "role": "assistant"},
        ],
        "rejected": [
            {"content": prompt, "role": "user"},
            {"content": rejected, "role": "assistant"},
        ],
    }


def map_to_standard_format_json(x):
    chosen = x["chosen"][1]["content"]
    rejected = x["rejected"][1]["content"]
    prompt = x["chosen"][0]["content"]

    return {
        "prompt": [{"content": prompt, "role": "user"}],
        "chosen": [
            {"content": prompt, "role": "user"},
            {"content": chosen, "role": "assistant"},
        ],
        "rejected": [
            {"content": prompt, "role": "user"},
            {"content": rejected, "role": "assistant"},
        ],
    }


def get_dataset(
    dataset_name: str, subset_size: Optional[int] = None
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Returns (train_dataset, eval_dataset) where both have 'prompt','chosen','rejected'.
    """
    if dataset_name == "chatbot_arena_2024":
        ds_any = load_dataset(
            "allenai/tulu-2.5-preference-data", split="chatbot_arena_2024"
        )
    else:
        ds_any = load_dataset(dataset_name)

    ds: Dataset
    eval_ds: Optional[Dataset] = None

    # Select a reasonable split
    if isinstance(ds_any, DatasetDict):
        if "train_prefs" in ds_any:
            ds = ds_any["train_prefs"]  # e.g., UltraFeedback-binarized
            eval_ds = ds_any.get("test_prefs")
        elif "train" in ds_any:
            ds = ds_any["train"]
            eval_ds = ds_any.get("test")
        else:
            # Fallback: take the first split as train, second as test if available
            keys = list(ds_any.keys())
            ds = ds_any[keys[0]]
            if len(keys) > 1:
                eval_ds = ds_any[keys[1]]
    else:
        # It's already a Dataset (or split)
        ds = ds_any  # type: ignore

    # Optional sub-sample
    if subset_size is not None:
        ds = ds.select(range(min(subset_size, len(ds))))

    # Apply dataset-specific mappings
    if dataset_name == "HuggingFaceH4/ultrafeedback_binarized":

        def _map_ultra(x):
            return {
                "prompt": x["chosen"][:1],
                "chosen": x["chosen"],
                "rejected": x["rejected"],
            }

        ds = ds.map(_map_ultra)
        if eval_ds:
            eval_ds = eval_ds.map(_map_ultra)

    elif dataset_name == "trl-lib/hh-rlhf-helpful-base":

        def _map_hh(x):
            return {
                "prompt": x["prompt"],
                "chosen": x["prompt"] + x["chosen"],
                "rejected": x["prompt"] + x["rejected"],
            }

        ds = ds.map(_map_hh)
        if eval_ds:
            eval_ds = eval_ds.map(_map_hh)

    elif dataset_name == "Intel/orca_dpo_pairs":
        # rename 'question' to 'prompt' then standard format
        ds = ds.map(
            lambda x: {
                "prompt": x["question"],
                "chosen": x["chosen"],
                "rejected": x["rejected"],
            }
        )
        ds = ds.map(map_to_standard_format)
        # Note: Intel/orca_dpo_pairs usually only has train, so eval_ds might be None

    elif dataset_name == "chatbot_arena_2024":
        ds = ds.map(map_to_standard_format_json)

    # Ensure required columns and remove extras
    needed = {"prompt", "chosen", "rejected"}
    missing = [c for c in needed if c not in ds.column_names]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' must have columns {needed}. Missing: {missing}"
        )

    ds = ds.remove_columns([c for c in ds.column_names if c not in needed])
    if eval_ds is not None:
        eval_ds = eval_ds.remove_columns(
            [c for c in eval_ds.column_names if c not in needed]
        )

    return ds, eval_ds
