import os
import sys
import types
import argparse
import gc
import random
import pickle
import string
import time
import torch
import math
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset, concatenate_datasets
from src.dataset_utils import get_dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainerCallback,
)
from src.train_utils import (
    compute_avg_logprob,
    compute_preference_accuracy,
    get_member_indices,
    compute_metrics,
    quick_mia_score,
    SamplingScheme,
)
from src.shadow_train_logging import (
    EpochMetricsLoggerCallback,
    log_phase_duration,
    parse_bool_flag,
    print_epoch_metric_summary,
    summarize_epoch_metrics,
)
from peft import (
    LoraConfig,
    VeraConfig,
    AdaLoraConfig,
    RandLoraConfig,
    ShiraConfig,
    DeloraConfig,
)

from src.trainers import (
    build_sft_trainer,
    build_dpo_trainer,
    build_ppo_trainer,
    build_grpo_trainer,
    build_kto_trainer,
    build_orpo_trainer,
    build_reward_trainer,
)
from src.lora_dp_utils import (
    partition_dataset_into_k_groups,
    extract_lora_state,
    load_lora_state,
    project_lora_state_toward_init,
    aggregate_lora_states,
    gaussian_std_for_mu_dp,
    add_gaussian_noise_to_lora_state,
)

try:
    import torch.distributed as dist
except Exception:
    dist = None

ROOT = str(REPO_ROOT)


def _sanitize_sys_path(*, drop_paths: tuple[str, ...] = ()) -> list[dict[str, str]]:
    removed: list[dict[str, str]] = []
    cleaned: list[Any] = []
    seen: set[str] = set()
    normalized_drop_paths: set[str] = set()
    for path in drop_paths:
        expanded = os.path.expanduser(path)
        normalized_drop_paths.add(expanded if os.path.isabs(expanded) else os.path.abspath(expanded))
    for entry in sys.path:
        key = entry if isinstance(entry, str) else repr(entry)
        if key in seen:
            continue
        seen.add(key)
        if entry == "":
            cleaned.append(entry)
            continue
        if not isinstance(entry, str):
            cleaned.append(entry)
            continue
        expanded_entry = os.path.expanduser(entry)
        normalized_entry = expanded_entry if os.path.isabs(expanded_entry) else os.path.abspath(expanded_entry)
        if normalized_entry in normalized_drop_paths:
            removed.append({"path": entry, "error": "dropped bootstrap path after import"})
            continue
        try:
            os.path.exists(entry)
        except OSError as exc:
            removed.append({"path": entry, "error": str(exc)})
            continue
        cleaned.append(entry)
    if cleaned != sys.path:
        sys.path[:] = cleaned
    for item in removed:
        sys.path_importer_cache.pop(item["path"], None)
    return removed


_REMOVED_IMPORT_PATHS = _sanitize_sys_path(drop_paths=(ROOT,))

slurm_job_id: int = os.getenv("SLURM_JOB_ID")


def compute_total_step(
    num_samples: int,
    per_device_bs: int,
    grad_acc: int,
    num_epochs: int,
    world_size: int,
) -> int:
    steps_per_epoch_dataloader = math.ceil(num_samples / (per_device_bs * world_size))
    steps_per_epoch_optimizer = math.ceil(steps_per_epoch_dataloader / grad_acc)
    return num_epochs * steps_per_epoch_optimizer


def _ensure_token_type_ids_for_gemma3(model):
    if getattr(model, "_codex_token_type_ids_wrapped", False):
        return model
    if getattr(model.config, "model_type", None) != "gemma3":
        return model

    orig_forward = model.forward

    def wrapped_forward(self, *args, **kwargs):
        if kwargs.get("token_type_ids", None) is None:
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            if input_ids is not None:
                kwargs["token_type_ids"] = torch.zeros_like(input_ids)
            else:
                inputs_embeds = kwargs.get("inputs_embeds", None)
                if inputs_embeds is not None:
                    kwargs["token_type_ids"] = torch.zeros(
                        inputs_embeds.shape[:2],
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
        return orig_forward(*args, **kwargs)

    model.forward = types.MethodType(wrapped_forward, model)
    model._codex_token_type_ids_wrapped = True
    return model


def _unwrap_model_for_post_training_eval(model, training_method: str):
    if training_method != "ppo" or model is None:
        return model

    current = model
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        policy = getattr(current, "policy", None)
        if policy is not None:
            return policy
        current = getattr(current, "module", None)

    print(
        "[*] Warning: could not unwrap PPO policy model for eval; "
        "falling back to trainer.model."
    )
    return model


if slurm_job_id is None:
    slurm_job_id: int = 0


def _get_dist_info():
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _random_string_of_length(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    if length <= 0:
        return ""
    return "".join(rng.choice(alphabet) for _ in range(length))


def _make_preference_sample(prompt: str, chosen: str, rejected: str) -> Dict[str, Any]:
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


def _build_special_canaries(
    base_ds: Dataset,
    *,
    num_normal: int,
    num_random_string: int,
    num_mislabeled: int,
    num_random_question: int,
    num_reasonable_question: int,
    num_reasonable_question_random_chosen: int,
    seed: int,
) -> Tuple[Dataset, Dict[str, Any]]:
    # Canary variants:
    # - normal_canaries: copied real sample without changes
    # - random_strings: random prompt, random chosen, random rejected
    # - mislabeled: swap chosen and rejected from a real sample
    # - random_question: random prompt with real chosen/rejected responses
    # - reasonable_question: real prompt with random chosen/rejected responses
    # - reasonable_question_random_chosen: real prompt with random chosen and real rejected
    counts = {
        "normal_canaries": int(num_normal),
        "random_strings": int(num_random_string),
        "mislabeled": int(num_mislabeled),
        "random_question": int(num_random_question),
        "reasonable_question": int(num_reasonable_question),
        "reasonable_question_random_chosen": int(
            num_reasonable_question_random_chosen
        ),
    }
    total = sum(counts.values())
    metadata = {
        "base_dataset_size": len(base_ds),
        "augmented_dataset_size": len(base_ds) + total,
        "counts": counts,
        "items": [],
        "indices": [],
        "kinds": [],
        "source_indices": [],
        "seed": seed,
    }
    if total == 0:
        return base_ds, metadata
    if len(base_ds) == 0:
        raise ValueError("Cannot add canaries to an empty dataset.")

    rng = random.Random(seed)
    canary_rows: List[Dict[str, Any]] = []

    def sample_source_index() -> int:
        return rng.randrange(len(base_ds))

    def add_item(kind: str, sample: Dict[str, Any], source_index: Optional[int]) -> None:
        location = len(base_ds) + len(canary_rows)
        metadata["items"].append(
            {
                "index": location,
                "kind": kind,
                "source_index": source_index,
            }
        )
        canary_rows.append(sample)

    for _ in range(counts["normal_canaries"]):
        src_idx = sample_source_index()
        src = base_ds[src_idx]
        add_item(
            "normal_canaries",
            _make_preference_sample(
                prompt=src["prompt"][0]["content"],
                chosen=src["chosen"][1]["content"],
                rejected=src["rejected"][1]["content"],
            ),
            src_idx,
        )

    for _ in range(counts["random_strings"]):
        src_idx = sample_source_index()
        src = base_ds[src_idx]
        add_item(
            "random_strings",
            _make_preference_sample(
                prompt=_random_string_of_length(
                    rng,
                    len(src["prompt"][0]["content"]),
                ),
                chosen=_random_string_of_length(
                    rng,
                    len(src["chosen"][1]["content"]),
                ),
                rejected=_random_string_of_length(
                    rng,
                    len(src["rejected"][1]["content"]),
                ),
            ),
            src_idx,
        )

    for _ in range(counts["mislabeled"]):
        src_idx = sample_source_index()
        src = base_ds[src_idx]
        add_item(
            "mislabeled",
            _make_preference_sample(
                prompt=src["prompt"][0]["content"],
                chosen=src["rejected"][1]["content"],
                rejected=src["chosen"][1]["content"],
            ),
            src_idx,
        )

    for _ in range(counts["random_question"]):
        src_idx = sample_source_index()
        src = base_ds[src_idx]
        add_item(
            "random_question",
            _make_preference_sample(
                prompt=_random_string_of_length(
                    rng,
                    len(src["prompt"][0]["content"]),
                ),
                chosen=src["chosen"][1]["content"],
                rejected=src["rejected"][1]["content"],
            ),
            src_idx,
        )

    for _ in range(counts["reasonable_question"]):
        src_idx = sample_source_index()
        src = base_ds[src_idx]
        add_item(
            "reasonable_question",
            _make_preference_sample(
                prompt=src["prompt"][0]["content"],
                chosen=_random_string_of_length(
                    rng,
                    len(src["chosen"][1]["content"]),
                ),
                rejected=_random_string_of_length(
                    rng,
                    len(src["rejected"][1]["content"]),
                ),
            ),
            src_idx,
        )

    for _ in range(counts["reasonable_question_random_chosen"]):
        src_idx = sample_source_index()
        src = base_ds[src_idx]
        add_item(
            "reasonable_question_random_chosen",
            _make_preference_sample(
                prompt=src["prompt"][0]["content"],
                chosen=_random_string_of_length(
                    rng,
                    len(src["chosen"][1]["content"]),
                ),
                rejected=src["rejected"][1]["content"],
            ),
            src_idx,
        )

    metadata["indices"] = [item["index"] for item in metadata["items"]]
    metadata["kinds"] = [item["kind"] for item in metadata["items"]]
    metadata["source_indices"] = [item["source_index"] for item in metadata["items"]]
    augmented_ds = concatenate_datasets([base_ds, Dataset.from_list(canary_rows)])
    return augmented_ds, metadata


def _build_training_save_path(args: argparse.Namespace, *, epoch_subdir: str | None = None) -> str:
    if args.use_peft:
        save_path = os.path.join(
            args.output_dir,
            f"{args.dataset_name.split('/')[-1]}",
            f"{args.method}",
            f"{args.peft_version}",
            f"{args.model_name.split('/')[-1]}",
        )
    else:
        save_path = os.path.join(
            args.output_dir,
            f"{args.dataset_name.split('/')[-1]}",
            f"{args.method}",
            f"{args.model_name.split('/')[-1]}",
        )
    if args.subdir is not None:
        save_path = os.path.join(save_path, f"{args.subdir}")
    if epoch_subdir is not None:
        save_path = os.path.join(save_path, epoch_subdir)
    return save_path


def _compute_eval_summaries(loss_records: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    utility_metrics: Dict[str, float] = {}
    train_acc, test_acc = compute_preference_accuracy(loss_records)
    utility_metrics["train_acc"] = train_acc
    utility_metrics["test_acc"] = test_acc

    avg_chosen, avg_rejected = compute_avg_logprob(loss_records)
    utility_metrics["avg_logprob_chosen"] = avg_chosen
    utility_metrics["avg_logprob_rejected"] = avg_rejected

    labels = torch.as_tensor(loss_records["is_member"], dtype=torch.bool).reshape(-1)
    if labels.numel() == 0 or labels.unique().numel() < 2:
        return utility_metrics, None

    return utility_metrics, quick_mia_score(loss_records)


def main(args=None):
    total_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["sft", "dpo", "ppo", "grpo", "kto", "orpo", "reward", "none"],
    )
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--load_reward_model", type=str, default=None)

    ap.add_argument("--num_shadows", type=int, default=16)
    ap.add_argument("--shadow_id", type=int, required=True)
    ap.add_argument("--shadow_seed", type=int, default=123)

    ap.add_argument("--do_eval", type=str, default=True)
    ap.add_argument(
        "--log_metrics_each_epoch",
        action="store_true",
        help=(
            "Print and persist the trainer's epoch-level metrics. This enables "
            "epoch-end evaluation when an eval split is available."
        ),
    )
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--train_samples", type=int, default=2000)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument(
        "--num_normal_canaries",
        type=int,
        default=0,
        help="Number of canaries that copy a real sample without changes.",
    )
    ap.add_argument(
        "--num_random_string_canaries",
        type=int,
        default=0,
        help="Number of canaries with random prompt/chosen/rejected strings.",
    )
    ap.add_argument(
        "--num_mislabeled_canaries",
        type=int,
        default=0,
        help="Number of canaries that swap chosen and rejected responses.",
    )
    ap.add_argument(
        "--num_random_question_canaries",
        type=int,
        default=0,
        help="Number of canaries with random prompts and real responses.",
    )
    ap.add_argument(
        "--num_reasonable_question_canaries",
        type=int,
        default=0,
        help="Number of canaries with real prompts and random responses.",
    )
    ap.add_argument(
        "--num_reasonable_question_random_chosen_canaries",
        type=int,
        default=0,
        help=(
            "Number of canaries with real prompts, a random chosen response, and "
            "the original rejected response."
        ),
    )
    ap.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optimizer name for HF TrainingArguments (e.g., paged_adamw_8bit).",
    )
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument(
        "--use_zero1",
        action="store_true",
        help="Enable DeepSpeed ZeRO-1 with DDP.",
    )
    ap.add_argument(
        "--use_zero2",
        action="store_true",
        help="Enable DeepSpeed ZeRO-2 with DDP.",
    )
    ap.add_argument(
        "--use_zero3",
        action="store_true",
        help="Enable DeepSpeed ZeRO-3 with DDP.",
    )
    ap.add_argument(
        "--use_zero0",
        action="store_true",
        help="Enable DeepSpeed ZeRO-0 with DDP.",
    )

    # GPRO-specific
    ap.add_argument("--grpo_num_generations", type=int, default=2)
    ap.add_argument(
        "--grpo_loss_type",
        type=str,
        default="dapo",
        choices=["dapo", "grpo", "dr_grpo", "bnpo"],
    )
    ap.add_argument("--grpo_beta", type=float, default=0.0)
    ap.add_argument(
        "--grpo_importance_sampling_level",
        type=str,
        default="token",
        choices=["token", "sequence"],
    )
    ap.add_argument("--grpo_epsilon", type=float, default=0.2)
    # DPO-specific
    ap.add_argument("--dpo_loss_type", type=str, default="sigmoid")
    ap.add_argument("--dpo_beta", type=float, default=0.1)

    # SFT-specific
    ap.add_argument("--sft_loss_type", type=str, default="nll", choices=["nll", "dft"])

    # KTO-specific
    ap.add_argument("--kto_beta", type=float, default=0.1)

    # ORPO-specific
    ap.add_argument("--orpo_beta", type=float, default=0.1)

    # Peft-Specific
    ap.add_argument("--use_peft", action="store_true")
    ap.add_argument(
        "--peft_version",
        type=str,
        default="lora",
        choices=[
            "lora",
            "rslora",
            "dora",
            "vera",
            "adalora",
            "randlora",
            "shira",
            "delora",
        ],
        help="Adapter family to use when --use_peft is enabled.",
    )
    ap.add_argument("--peft_param_a", type=int, default=8)
    ap.add_argument("--peft_param_b", type=int, default=32)

    # LoRA-specific
    ap.add_argument(
        "--enable_lora_ensemble",
        action="store_true",
        help=(
            "Enable K-group LoRA ensemble training with clipping/projection to shared "
            "initialization and Gaussian-noised aggregation."
        ),
    )
    ap.add_argument(
        "--lora_num_groups",
        type=int,
        default=4,
        help="Number of disjoint groups for LoRA ensemble training.",
    )
    ap.add_argument(
        "--lora_clip_factor",
        type=float,
        default=1.0,
        help="L2 clipping radius for LoRA updates around shared initialization.",
    )
    ap.add_argument(
        "--lora_mu_dp",
        type=float,
        default=1.0,
        help="Target mu-DP level for Gaussian noise calibration.",
    )
    ap.add_argument(
        "--lora_noise_seed",
        type=int,
        default=12345,
        help="Seed for Gaussian noise added to aggregated LoRA parameters.",
    )

    ap.add_argument("--output_dir", type=str, default="out/models/")
    ap.add_argument("--subdir", type=str, default=None)
    ap.add_argument("--log_to_wandb", action="store_true")
    ap.add_argument(
        "--save_epoch_snapshots",
        action="store_true",
        help=(
            "After each non-final epoch, score the current shadow model and save a "
            "standard shadow pickle under an `epoch_XXX/` subdirectory."
        ),
    )
    ap.add_argument(
        "--base_num_train_epochs",
        type=int,
        default=None,
        help="Original standard-HP epoch count used to derive a long-epochs run.",
    )
    ap.add_argument(
        "--target_num_train_epochs",
        type=int,
        default=None,
        help="Planned final epoch count for a long-epochs run.",
    )
    args = ap.parse_args(args=args)
    if args.enable_lora_ensemble:
        args.use_peft = True
    args.epoch_snapshot = False
    if args.save_epoch_snapshots and args.enable_lora_ensemble:
        print(
            "[!] --save_epoch_snapshots is not supported with --enable_lora_ensemble. "
            "Disabling epoch snapshots for this run.",
            flush=True,
        )
        args.save_epoch_snapshots = False
    if (
        args.save_epoch_snapshots
        or args.base_num_train_epochs is not None
        or args.target_num_train_epochs is not None
    ):
        if args.base_num_train_epochs is None:
            args.base_num_train_epochs = int(args.num_train_epochs)
        if args.target_num_train_epochs is None:
            args.target_num_train_epochs = int(args.num_train_epochs)

    zero_flags = [args.use_zero0, args.use_zero1, args.use_zero2, args.use_zero3]
    if sum(int(flag) for flag in zero_flags) > 1:
        raise ValueError(
            "Please enable only one of use_zero0/use_zero1/use_zero2/use_zero3."
        )
    if args.use_zero0:
        zero_stage = 0
    elif args.use_zero1:
        zero_stage = 1
    elif args.use_zero2:
        zero_stage = 2
    elif args.use_zero3:
        zero_stage = 3
    else:
        zero_stage = None
    args.zero_stage = zero_stage

    setup_start = time.perf_counter()
    args.do_eval = parse_bool_flag(args.do_eval)

    print("\n[*] Training Configuration:")
    for k, v in vars(args).items():
        print(f"{k:30}: {v}")
    print("")

    random.seed(args.shadow_seed)
    torch.manual_seed(args.shadow_seed)

    if args.log_to_wandb:
        import wandb

        wandb_kwargs = dict(
            name=f"shadow_{args.shadow_id}_{args.method}_{args.dataset_name.split('/')[-1]}_{args.model_name.split('/')[-1]}",
            job_type="training",
            config={
                **vars(args),
                "SLURM_JOB_ID": slurm_job_id,
            },
        )
        wandb_project = os.environ.get("WANDB_PROJECT")
        if wandb_project:
            wandb_kwargs["project"] = wandb_project
        wandb.init(**wandb_kwargs)
    log_phase_duration(
        "argument parsing, seeding, and optional wandb init",
        time.perf_counter() - setup_start,
    )

    data_start = time.perf_counter()
    if args.method.lower() != "reward":  # Reward model trains over the whole dataset
        ds, _ = get_dataset(
            dataset_name=args.dataset_name,
            subset_size=args.train_samples,
        )
        ds, special_canary_meta = _build_special_canaries(
            ds,
            num_normal=args.num_normal_canaries,
            num_random_string=args.num_random_string_canaries,
            num_mislabeled=args.num_mislabeled_canaries,
            num_random_question=args.num_random_question_canaries,
            num_reasonable_question=args.num_reasonable_question_canaries,
            num_reasonable_question_random_chosen=
                args.num_reasonable_question_random_chosen_canaries,
            seed=args.shadow_seed,
        )
        NUM_SAMPLES = len(ds)
        print(
            f"\n[*] Loaded dataset '{args.dataset_name}' with {NUM_SAMPLES} samples.\n"
        )

        member_ids = get_member_indices(
            num_samples=NUM_SAMPLES,
            num_shadows=args.num_shadows,
            shadow_id=args.shadow_id,
            seed=args.shadow_seed,
        )

        member_set = set(member_ids)
        nonmember_ids = sorted(set(range(NUM_SAMPLES)) - member_set)
        train_ds = ds.select(member_ids)
        eval_ds = ds.select(nonmember_ids)
        membership_mask = [int(i in member_set) for i in range(NUM_SAMPLES)]
    else:
        ds = get_dataset(
            dataset_name=args.dataset_name,
            subset_size=args.train_samples,
        )
        train_ds, eval_ds = ds
        if eval_ds is None:
            args.do_eval = False
        NUM_SAMPLES = len(train_ds)
        print(
            f"\n[*] Loaded dataset '{args.dataset_name}' with {NUM_SAMPLES} samples.\n"
        )
    log_phase_duration("dataset loading and split construction", time.perf_counter() - data_start)

    if args.log_metrics_each_epoch or args.save_epoch_snapshots:
        if eval_ds is None or len(eval_ds) == 0:
            print(
                "[!] Epoch-end logging or snapshots were requested, but no eval split "
                "is available. Epoch-end evaluation will stay disabled.",
                flush=True,
            )
            args.save_epoch_snapshots = False
        elif not args.do_eval:
            print(
                "[*] Enabling --do_eval because epoch-end logging or snapshots were requested.",
                flush=True,
            )
            args.do_eval = True

    tokenizer_start = time.perf_counter()
    if args.model_name in ["Qwen/Qwen3-VL-2B-Instruct", "google/gemma-3-4b-it"]:
        print(
            f"[!] Multimodal Detected: {args.model_name} | Extracting Tokenizer from AutoProcessor."
        )
        processor = AutoProcessor.from_pretrained(
            args.model_name, use_fast=True, trust_remote_code=True
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, padding_side="left"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.chat_template = """{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'] + eos_token }}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|assistant|>\n' }}
    {%- endif %}"""
    log_phase_duration("tokenizer / processor load", time.perf_counter() - tokenizer_start)

    num_gpus = torch.cuda.device_count()
    if args.zero_stage is not None:
        device_map = None  # DeepSpeed handles device placement for ZeRO stages
    elif num_gpus > 1:  # Cluster workaround for multi-GPU placement.
        from accelerate import Accelerator

        device_index = Accelerator().process_index
        device_map = {"": device_index}
    else:
        device_map = "auto"

    if args.method.lower() != "reward":
        model_load_start = time.perf_counter()

        torch_dtype = (
            torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else "auto"
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model = _ensure_token_type_ids_for_gemma3(model)
        model.train()
        model.config.use_cache = False  # MiniCPM fix
        log_phase_duration("policy model load", time.perf_counter() - model_load_start)

    if (
        args.method.lower() == "grpo" or args.method.lower() == "ppo"
    ) and args.load_reward_model is not None:
        reward_model_start = time.perf_counter()
        print(f"[*] Loading Reward Model from {args.load_reward_model} for GRPO/PPO...")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.load_reward_model,
            device_map="cpu",
            num_labels=1,
            dtype="auto",
            trust_remote_code=True,
        )
        reward_model.to("cuda")
        reward_model.eval()
        reward_model.config.use_cache = False  # MiniCPM fix
        log_phase_duration("reward model load", time.perf_counter() - reward_model_start)

    if args.enable_lora_ensemble:
        if args.method.lower() == "none":
            raise ValueError(
                "--enable_lora_ensemble cannot be used with --method none."
            )
        if args.lora_num_groups < 1:
            raise ValueError(
                "--lora_num_groups must be >= 1 when lora ensemble is enabled."
            )
        if args.lora_clip_factor <= 0:
            raise ValueError(
                "--lora_clip_factor must be > 0 when lora ensemble is enabled."
            )
        if args.lora_mu_dp <= 0:
            raise ValueError("--lora_mu_dp must be > 0 when lora ensemble is enabled.")
        if len(train_ds) < args.lora_num_groups:
            raise ValueError(
                f"train dataset size ({len(train_ds)}) must be >= "
                f"lora_num_groups ({args.lora_num_groups})."
            )

    if args.use_peft:
        if args.peft_version in ["lora", "rslora", "dora"]:
            peft_config = LoraConfig(
                r=args.peft_param_a,
                lora_alpha=args.peft_param_b,
                target_modules="all-linear",
                use_rslora=True if args.peft_version == "rslora" else False,
                use_dora=True if args.peft_version == "dora" else False,
            )
            adapter_param_patterns = ("lora_",)
        elif args.peft_version == "vera":
            peft_config = VeraConfig(
                r=(args.peft_param_a if args.peft_param_a > 256 else 256),
                target_modules="all-linear",
            )
            adapter_param_patterns = ("vera_",)
        elif args.peft_version == "adalora":
            rank, world_size = _get_dist_info()
            total_step = compute_total_step(
                num_samples=len(train_ds),
                per_device_bs=args.per_device_train_batch_size,
                grad_acc=args.gradient_accumulation_steps,
                num_epochs=args.num_train_epochs,
                world_size=world_size,
            )
            peft_config = AdaLoraConfig(
                target_r=8,
                init_r=32,
                deltaT=1,
                target_modules="all-linear",
                total_step=total_step,
            )
            adapter_param_patterns = ("adalora_",)
        elif args.peft_version == "randlora":
            peft_config = RandLoraConfig(
                r=args.peft_param_a,
                randlora_alpha=(args.peft_param_a * 20),
                sparse=False,
                very_sparse=False,
            )
            adapter_param_patterns = ("randlora_",)
        elif args.peft_version == "shira":
            peft_config = ShiraConfig(
                r=(args.peft_param_a if args.peft_param_a > 32 else 32),
            )
            adapter_param_patterns = ("shira_",)
        elif args.peft_version == "delora":
            peft_config = DeloraConfig(
                r=args.peft_param_a,
                target_modules="all-linear",
            )
            adapter_param_patterns = ("delora_",)
        else:
            raise ValueError(f"Unknown lora_version: {args.peft_version}")
    else:
        peft_config = None
        adapter_param_patterns = (
            "lora_",
            "vera_",
            "adalora_",
            "randlora_",
            "shira_",
            "delora_",
        )

    def build_ds_config(args, zero_stage):
        micro_bs = args.per_device_train_batch_size
        grad_acc = args.gradient_accumulation_steps

        zero_optimization = {
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        }

        if zero_stage >= 2:
            zero_optimization.update(
                {
                    "reduce_scatter": True,
                    "allgather_partitions": True,
                    "reduce_bucket_size": 5e7,
                }
            )

        if zero_stage == 3:
            zero_optimization.update(
                {
                    "stage3_prefetch_bucket_size": 5e7,
                    "stage3_param_persistence_threshold": 1e6,
                    "stage3_gather_16bit_weights_on_model_save": False,
                }
            )

        return {
            "train_micro_batch_size_per_gpu": micro_bs,
            "gradient_accumulation_steps": grad_acc,
            "zero_optimization": zero_optimization,
            "bf16": {"enabled": bool(args.bf16)},
            "fp16": {"enabled": bool(args.fp16)},
            "gradient_clipping": "auto",
            "steps_per_print": 2000,
            "wall_clock_breakdown": False,
        }

    deepspeed_cfg = (
        build_ds_config(args, args.zero_stage) if args.zero_stage is not None else None
    )

    def _build_trainer_for_method(
        method,
        model_or_name_in,
        train_in_subset,
        eval_in_subset,
        do_eval_in,
        peft_config_in,
    ):
        if method == "sft":
            return build_sft_trainer(
                model_name=model_or_name_in,
                tokenizer=tokenizer,
                train_in=train_in_subset,
                eval_in=eval_in_subset,
                do_eval=do_eval_in,
                max_length=args.max_seq_len,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                lr=args.lr,
                bf16=True if args.bf16 else False,
                fp16=True if args.fp16 else False,
                logging_tool="wandb" if args.log_to_wandb else "none",
                loss_type=args.sft_loss_type,
                peft_config=peft_config_in,
                ds_config=deepspeed_cfg,
            )
        if method == "dpo":
            if (
                hasattr(train_in_subset, "column_names")
                and "images" not in train_in_subset.column_names
                and hasattr(model_or_name_in, "config")
            ):
                if getattr(model_or_name_in.config, "model_type", None) != "text-only":
                    print(
                        "[!] Vision-capable model detected with text-only data; "
                        "forcing text-only processing for DPO."
                    )
                model_or_name_in.config.model_type = "text-only"
            return build_dpo_trainer(
                model_or_name=model_or_name_in,
                tokenizer=tokenizer,
                train_in=train_in_subset,
                eval_in=eval_in_subset,
                do_eval=do_eval_in,
                max_length=args.max_seq_len,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                lr=args.lr,
                bf16=True if args.bf16 else False,
                fp16=True if args.fp16 else False,
                logging_tool="wandb" if args.log_to_wandb else "none",
                loss_type=args.dpo_loss_type,
                beta=args.dpo_beta,
                peft_config=peft_config_in,
                optim=args.optim,
                ds_config=deepspeed_cfg,
            )
        if method == "ppo":
            return build_ppo_trainer(
                model_name=model_or_name_in,
                reward_model_id=(
                    args.model_name if args.load_reward_model is None else reward_model
                ),
                tokenizer=tokenizer,
                train_in=train_in_subset,
                eval_in=eval_in_subset,
                do_eval=do_eval_in,
                max_length=args.max_seq_len,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                lr=args.lr,
                bf16=True if args.bf16 else False,
                fp16=True if args.fp16 else False,
                logging_tool="wandb" if args.log_to_wandb else "none",
                peft_config=peft_config_in,
                ds_config=deepspeed_cfg,
            )
        if method == "grpo":
            return build_grpo_trainer(
                model_name=model_or_name_in,
                reward_model_id=(
                    args.model_name if args.load_reward_model is None else reward_model
                ),
                tokenizer=tokenizer,
                train_in=train_in_subset,
                eval_in=eval_in_subset,
                do_eval=do_eval_in,
                max_length=args.max_seq_len,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_generations=args.grpo_num_generations,
                num_train_epochs=args.num_train_epochs,
                lr=args.lr,
                bf16=True if args.bf16 else False,
                fp16=True if args.fp16 else False,
                logging_tool="wandb" if args.log_to_wandb else "none",
                loss_type=args.grpo_loss_type,
                beta=args.grpo_beta,
                importance_sampling_level=args.grpo_importance_sampling_level,
                epsilon=args.grpo_epsilon,
                peft_config=peft_config_in,
                ds_config=deepspeed_cfg,
            )
        if method == "kto":
            return build_kto_trainer(
                model_name=model_or_name_in,
                tokenizer=tokenizer,
                train_in=train_in_subset,
                eval_in=eval_in_subset,
                do_eval=do_eval_in,
                max_length=args.max_seq_len,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                lr=args.lr,
                bf16=True if args.bf16 else False,
                fp16=True if args.fp16 else False,
                logging_tool="wandb" if args.log_to_wandb else "none",
                beta=args.kto_beta,
                peft_config=peft_config_in,
                ds_config=deepspeed_cfg,
            )
        if method == "orpo":
            return build_orpo_trainer(
                model_name=model_or_name_in,
                tokenizer=tokenizer,
                train_in=train_in_subset,
                eval_in=eval_in_subset,
                do_eval=do_eval_in,
                max_length=args.max_seq_len,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                lr=args.lr,
                bf16=True if args.bf16 else False,
                fp16=True if args.fp16 else False,
                logging_tool="wandb" if args.log_to_wandb else "none",
                beta=args.orpo_beta,
                peft_config=peft_config_in,
                ds_config=deepspeed_cfg,
            )
        if method == "reward":
            return build_reward_trainer(
                model_name=model_or_name_in,
                tokenizer=tokenizer,
                train_in=train_in_subset,
                eval_in=eval_in_subset,
                do_eval=do_eval_in,
                max_length=args.max_seq_len,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                lr=args.lr,
                bf16=True if args.bf16 else False,
                fp16=True if args.fp16 else False,
                logging_tool="wandb" if args.log_to_wandb else "none",
                peft_config=peft_config_in,
            )
        if method == "none":
            return None
        raise ValueError(f"Unknown training method: {method}")

    schemes = [
        SamplingScheme(name="greedy", type="top_k", k=1, T=1.0),
        SamplingScheme(
            name="topk40_t1", type="top_k", k=40, T=1.0
        ),  # Paper's choice
        SamplingScheme(name="topp0.9_t1", type="top_p", p=0.9, T=1.0),
    ]

    def _evaluate_shadow_model(
        model_in,
        *,
        dataset_in,
        membership_mask_in,
        log_prefix: str,
        metrics_phase_name: str,
        log_to_wandb_in: bool,
        utility_wandb_prefix: str = "",
        mia_wandb_prefix: str = "mia_",
        allow_single_class_mia: bool = False,
        summary_label: str = "Utility (standard shadow universe)",
    ) -> Optional[dict[str, Any]]:
        metrics_start = time.perf_counter()
        loss_records = compute_metrics(
            ds=dataset_in,
            max_length=args.max_seq_len,
            model=model_in,
            tokenizer=tokenizer,
            membership_mask=membership_mask_in,
            extraction_schemes=schemes,
            np_p_list=(0.1, 0.5, 0.9, 0.99),
            gather_across_ranks=False if args.zero_stage == 3 else True,
        )
        log_phase_duration(metrics_phase_name, time.perf_counter() - metrics_start)
        rank, world_size = _get_dist_info()
        if loss_records is None:
            print(
                f"{log_prefix} Rank {rank}/{world_size} skipping metrics aggregation; "
                "waiting for rank 0.",
                flush=True,
            )
            if dist is not None and dist.is_available() and dist.is_initialized():
                dist.barrier()
            return None

        utility_metrics, mia_scores = _compute_eval_summaries(loss_records)

        if log_to_wandb_in:
            wandb.log({f"{utility_wandb_prefix}{k}": v for k, v in utility_metrics.items()})

        print(f"\n{log_prefix} {summary_label}:", flush=True)
        for k, v in utility_metrics.items():
            print(f"    {k}: {v:.4f}", flush=True)
        print("", flush=True)

        if mia_scores is None:
            if allow_single_class_mia:
                print(
                    f"{log_prefix} Membership Inference Attack Scores skipped "
                    "(dataset contains only one membership class).",
                    flush=True,
                )
                print("", flush=True)
            else:
                raise ValueError(
                    f"{log_prefix} expected both member and non-member samples for MIA summaries."
                )
        else:
            print(f"{log_prefix} Membership Inference Attack Scores:", flush=True)
            for k, v in mia_scores.items():
                print(f"    {k}: {v:.4f}", flush=True)
            print("", flush=True)

            if log_to_wandb_in:
                wandb.log({f"{mia_wandb_prefix}{k}": v for k, v in mia_scores.items()})

        return {
            "utility": utility_metrics,
            "loss_records": loss_records,
            "mia_scores": mia_scores,
        }

    def _serialize_shadow_output(
        *,
        model_in,
        log_prefix: str,
        metrics_phase_name: str,
        log_to_wandb_in: bool,
        epoch_subdir: str | None = None,
        path_name: str | None = None,
        args_override: Optional[Dict[str, Any]] = None,
        epoch_metrics_override: Optional[List[Dict[str, Any]]] = None,
        trainer_log_history_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        eval_artifacts = _evaluate_shadow_model(
            model_in,
            dataset_in=ds,
            membership_mask_in=membership_mask,
            log_prefix=log_prefix,
            metrics_phase_name=metrics_phase_name,
            log_to_wandb_in=log_to_wandb_in,
            summary_label="Utility (standard shadow universe)",
        )
        if eval_artifacts is None:
            return None

        out_args = dict(vars(args))
        if args_override:
            out_args.update(args_override)
        out = {
            "args": out_args,
            "utility": eval_artifacts["utility"],
            "membership_mask": membership_mask,
            "loss_records": eval_artifacts["loss_records"],
            "mia_scores": eval_artifacts["mia_scores"],
            "special_canaries": special_canary_meta,
        }
        if epoch_metrics_override is not None:
            out["epoch_metrics"] = epoch_metrics_override
        if trainer_log_history_override is not None:
            out["trainer_log_history"] = trainer_log_history_override

        save_path = _build_training_save_path(args, epoch_subdir=epoch_subdir)
        os.makedirs(save_path, exist_ok=True)
        final_path_name = path_name or f"shadow_{args.shadow_id}.pkl"
        save_start = time.perf_counter()
        with open(os.path.join(save_path, final_path_name), "wb") as f:
            pickle.dump(out, f)
        log_phase_duration(
            f"result serialization{'' if epoch_subdir is None else f' [{epoch_subdir}]'}",
            time.perf_counter() - save_start,
        )
        print(f"\n{log_prefix} Results saved to {save_path}.", flush=True)
        return save_path

    class EpochSnapshotExportCallback(TrainerCallback):
        def __init__(self) -> None:
            self.saved_epochs: set[int] = set()

        def on_evaluate(self, args_hf, state, control, **kwargs):
            if not args.save_epoch_snapshots:
                return control
            model_in = kwargs.get("model")
            if model_in is None:
                return control
            try:
                epoch_num = int(round(float(state.epoch)))
            except (TypeError, ValueError):
                return control
            target_epochs = int(args.target_num_train_epochs or args.num_train_epochs)
            if epoch_num <= 0 or epoch_num >= target_epochs or epoch_num in self.saved_epochs:
                return control
            self.saved_epochs.add(epoch_num)
            snapshot_subdir = f"epoch_{epoch_num:03d}"
            snapshot_start = time.perf_counter()
            print(
                f"[*] Exporting epoch snapshot {epoch_num}/{target_epochs - 1} "
                f"for shadow {args.shadow_id} into {snapshot_subdir}.",
                flush=True,
            )
            was_training = bool(getattr(model_in, "training", False))
            try:
                model_in.eval()
                _serialize_shadow_output(
                    model_in=model_in,
                    log_prefix=f"[*] Epoch snapshot {epoch_num} for Shadow {args.shadow_id}",
                    metrics_phase_name=f"compute_metrics() [epoch snapshot {epoch_num}]",
                    log_to_wandb_in=False,
                    epoch_subdir=snapshot_subdir,
                    args_override={
                        "num_train_epochs": int(epoch_num),
                        "epoch_snapshot": True,
                    },
                )
            finally:
                if was_training:
                    model_in.train()
            log_phase_duration(
                f"epoch snapshot {epoch_num} export",
                time.perf_counter() - snapshot_start,
            )
            return control

    def _attach_epoch_logger(trainer_in):
        if trainer_in is None:
            return trainer_in
        if args.log_metrics_each_epoch:
            trainer_in.add_callback(EpochMetricsLoggerCallback())
        if args.save_epoch_snapshots and training_method != "reward":
            trainer_in.add_callback(EpochSnapshotExportCallback())
        return trainer_in

    training_method = args.method.lower()
    trainer = None
    if args.enable_lora_ensemble:
        ensemble_train_start = time.perf_counter()
        grouped_train_ds = partition_dataset_into_k_groups(
            train_ds,
            num_groups=args.lora_num_groups,
            seed=args.shadow_seed,
        )
        clipped_group_states = []
        shared_init_lora_state = None
        model_for_groups = model if training_method != "reward" else args.model_name
        last_group_trainer = None

        print(
            f"[*] Running grouped LoRA ensemble for method='{training_method}' with "
            f"adapter='{args.peft_version}', {len(grouped_train_ds)} groups, "
            f"clip_factor={args.lora_clip_factor}, "
            f"mu_dp={args.lora_mu_dp}."
        )

        for gid, group_ds in enumerate(grouped_train_ds):
            group_train_start = time.perf_counter()
            print(
                f"[*] Group {gid + 1}/{len(grouped_train_ds)}: "
                f"{len(group_ds)} train samples."
            )
            group_trainer = _build_trainer_for_method(
                method=training_method,
                model_or_name_in=model_for_groups,
                train_in_subset=group_ds,
                eval_in_subset=eval_ds,
                do_eval_in=False,
                peft_config_in=peft_config if gid == 0 else None,
            )
            if group_trainer is None:
                raise ValueError(
                    "LoRA ensemble requires an actual training method, not 'none'."
                )

            _attach_epoch_logger(group_trainer)
            if shared_init_lora_state is None:
                shared_init_lora_state = extract_lora_state(
                    group_trainer.model, adapter_patterns=adapter_param_patterns
                )
            load_lora_state(group_trainer.model, shared_init_lora_state)
            group_trainer.train()

            trained_lora_state = extract_lora_state(
                group_trainer.model, adapter_patterns=adapter_param_patterns
            )
            clipped_state, l2_norm, proj_scale = project_lora_state_toward_init(
                trained_state=trained_lora_state,
                init_state=shared_init_lora_state,
                clip_factor=args.lora_clip_factor,
            )
            clipped_group_states.append(clipped_state)
            print(
                f"[*] Group {gid + 1}: LoRA update L2={l2_norm:.6f}, "
                f"projection_scale={proj_scale:.6f}"
            )
            log_phase_duration(
                f"LoRA ensemble group {gid + 1}/{len(grouped_train_ds)} train",
                time.perf_counter() - group_train_start,
            )

            model_for_groups = group_trainer.model
            if (
                last_group_trainer is not None
                and last_group_trainer is not group_trainer
            ):
                del last_group_trainer
            last_group_trainer = group_trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        aggregated_state = aggregate_lora_states(clipped_group_states)
        noise_std = gaussian_std_for_mu_dp(
            mu_dp=args.lora_mu_dp,
            clip_factor=args.lora_clip_factor,
            num_groups=len(clipped_group_states),
        )
        noisy_aggregated_state = add_gaussian_noise_to_lora_state(
            aggregated_state,
            noise_std=noise_std,
            seed=args.lora_noise_seed + args.shadow_id,
        )
        load_lora_state(model_for_groups, noisy_aggregated_state)
        model = model_for_groups
        trainer = last_group_trainer if training_method == "reward" else None
        if training_method != "reward" and last_group_trainer is not None:
            del last_group_trainer
        print(
            f"[*] Aggregated LoRA state loaded with Gaussian noise std={noise_std:.6f}."
        )
        log_phase_duration(
            "LoRA ensemble training and aggregation",
            time.perf_counter() - ensemble_train_start,
        )
    else:
        if training_method == "none":
            print("[*] No training method selected, skipping training.")
            model_load_start = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, trust_remote_code=True
            )
            trainer = None
            log_phase_duration("reference model load", time.perf_counter() - model_load_start)
        else:
            model_input = args.model_name if training_method == "reward" else model
            trainer = _build_trainer_for_method(
                method=training_method,
                model_or_name_in=model_input,
                train_in_subset=train_ds,
                eval_in_subset=eval_ds,
                do_eval_in=args.do_eval,
                peft_config_in=peft_config,
            )
            _attach_epoch_logger(trainer)
            train_start = time.perf_counter()
            trainer.train()
            log_phase_duration("trainer.train()", time.perf_counter() - train_start)

    print(f"[*] Training Complete for Shadow {args.shadow_id}.")
    if trainer is not None:
        model = _unwrap_model_for_post_training_eval(trainer.model, training_method)
        model = _ensure_token_type_ids_for_gemma3(model)
    trainer_log_history = []
    epoch_metrics = []
    if trainer is not None and args.log_metrics_each_epoch:
        trainer_log_history = list(getattr(trainer.state, "log_history", []) or [])
        epoch_metrics = summarize_epoch_metrics(trainer_log_history)
        print_epoch_metric_summary(epoch_metrics)

    if training_method != "reward":
        model.eval()
        save_path = _serialize_shadow_output(
            model_in=model,
            log_prefix=f"[*] Shadow {args.shadow_id}",
            metrics_phase_name="compute_metrics()",
            log_to_wandb_in=args.log_to_wandb,
            path_name="shadow_ref.pkl" if training_method == "none" else None,
            epoch_metrics_override=epoch_metrics if args.log_metrics_each_epoch else None,
            trainer_log_history_override=trainer_log_history if args.log_metrics_each_epoch else None,
        )
        if save_path is None:
            return
        log_phase_duration("total runtime", time.perf_counter() - total_start)
        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.barrier()

    else:
        if args.subdir is not None:
            save_path = os.path.join(
                args.output_dir,
                f"reward",
                f"{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{NUM_SAMPLES}",
                f"{args.subdir}",
            )
        else:
            save_path = os.path.join(
                args.output_dir,
                f"reward",
                f"{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{NUM_SAMPLES}",
            )
        os.makedirs(save_path, exist_ok=True)

        metrics = trainer_log_history or list(getattr(trainer.state, "log_history", []) or [])
        save_start = time.perf_counter()
        with open(os.path.join(save_path, f"metrics.pkl"), "wb") as f:
            pickle.dump(metrics, f)
        if args.log_metrics_each_epoch:
            with open(os.path.join(save_path, f"epoch_metrics.pkl"), "wb") as f:
                pickle.dump(epoch_metrics, f)
        model.save_pretrained(save_path)
        log_phase_duration("reward model save", time.perf_counter() - save_start)
        print(f"\n[*] Reward model saved to {save_path}.")
        log_phase_duration("total runtime", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
