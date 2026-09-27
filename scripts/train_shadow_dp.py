import os
import argparse
import random
import pickle
import string
import time
import torch
import sys
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset, concatenate_datasets
from src.dataset_utils import get_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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
from peft import LoraConfig
from src.trainers_dp import (
    build_sft_trainer,
    build_dpo_trainer,
    build_kto_trainer,
    build_orpo_trainer,
)
import dp_transformers

try:
    import torch.distributed as dist
except Exception:
    dist = None

slurm_job_id: int = os.getenv("SLURM_JOB_ID")
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


def pick_trainable_params(
    model: torch.nn.Module, ft_group: str, exclude_module: Optional[str]
) -> torch.nn.Module:
    model.requires_grad_(
        False
    )  # Freeze all parameters by default; we'll unfreeze the selected group below

    params = []
    layer_count = 0
    param_count = 0
    total_params = sum(param.numel() for param in model.parameters())

    for name, param in model.named_parameters():
        if "model.norm.weight" in name:
            params.append(name)
            param_count += param.numel()
            layer_count += 1
            param.requires_grad = True

        if exclude_module is None:
            if ft_group in name:
                params.append(name)
                param_count += param.numel()
                layer_count += 1
                param.requires_grad = True
        else:
            if ft_group in name and exclude_module not in name:
                params.append(name)
                param_count += param.numel()
                layer_count += 1
                param.requires_grad = True

    print(f"Total layers: {layer_count}")
    print(f"Total req. parameters: {param_count / 1e9:.2f}B")
    print(f"Total parameters (all): {total_params / 1e9:.2f}B\n")
    print(f"Total req. parameters (%): {param_count / total_params:.2%}")
    print("Preview List of selected parameters:")
    for name in params[:10]:
        print(name)
    print("...\n\n")
    return model


def main(args=None):
    total_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["sft", "dpo", "kto", "orpo"],
    )
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)

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
    # DP-specific
    ap.add_argument("--target_epsilon", type=float, default=10.0)
    ap.add_argument("--per_sample_max_grad_norm", type=float, default=1e-2)
    ap.add_argument(
        "--ghost_mode",
        action="store_true",
        help="Enable Opacus ghost clipping via GradSampleModuleFastGradientClipping.",
    )

    # DPO-specific
    ap.add_argument("--dpo_loss_type", type=str, default="sigmoid")
    ap.add_argument("--dpo_beta", type=float, default=0.1)

    # SFT-specific
    ap.add_argument("--sft_loss_type", type=str, default="nll", choices=["nll", "dft"])

    # KTO-specific
    ap.add_argument("--kto_beta", type=float, default=0.1)
    ap.add_argument(
        "--kto_z_clip",
        type=float,
        default=1.0,
        help=(
            "Per-channel clipping radius C_z for the cyclic shifted KTO z release."
        ),
    )
    ap.add_argument(
        "--kto_z_noise_multiplier",
        type=float,
        default=None,
        help=(
            "Optional noise multiplier sigma_z for the cyclic z release. If unset, "
            "the trainer calibrates the main DP-SGD noise jointly with z using a "
            "single compose-first/amplify-once accountant."
        ),
    )

    # ORPO-specific
    ap.add_argument("--orpo_beta", type=float, default=0.1)

    # LoRA-specific
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)

    # Selective Finetuning
    ap.add_argument("--enable_selective_finetuning", action="store_true")
    ap.add_argument(
        "--ft_group",
        type=str,
        default="none",
        choices=["attn", "mlp"],
        help="Which parameter group to finetune",
    )
    ap.add_argument(
        "--exclude_module",
        type=str,
        default=None,
        help="Module name substring to exclude from finetuning when selective finetuning is enabled",
    )

    ap.add_argument("--output_dir", type=str, default="out/models_dp/")
    ap.add_argument("--subdir", type=str, default=None)
    ap.add_argument("--log_to_wandb", action="store_true")
    args = ap.parse_args(args=args)

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
            name=f"dp_shadow_{args.shadow_id}_{args.method}_{args.dataset_name.split('/')[-1]}_{args.model_name.split('/')[-1]}",
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
    print(f"\n[*] Loaded dataset '{args.dataset_name}' with {NUM_SAMPLES} samples.\n")

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
    if args.log_metrics_each_epoch:
        if len(eval_ds) == 0:
            print(
                "[!] --log_metrics_each_epoch requested, but no eval split is available. "
                "Epoch-end evaluation will stay disabled.",
                flush=True,
            )
        elif not args.do_eval:
            print(
                "[*] Enabling --do_eval because --log_metrics_each_epoch was requested.",
                flush=True,
            )
            args.do_eval = True
    log_phase_duration("dataset loading and split construction", time.perf_counter() - data_start)

    tokenizer_start = time.perf_counter()
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
    log_phase_duration("tokenizer load", time.perf_counter() - tokenizer_start)

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
        model.train()
        model.config.use_cache = False  # MiniCPM fix
        log_phase_duration("policy model load", time.perf_counter() - model_load_start)

    if args.use_lora and args.enable_selective_finetuning:
        raise ValueError(
            "LoRA and selective finetuning cannot be enabled at the same time. Choose any one."
        )

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=32, target_modules="all-linear"
        )
    else:
        peft_config = None

    if args.enable_selective_finetuning:
        ft_group = args.ft_group
        exclude_module = args.exclude_module
        print(
            f"[*] Enabling selective finetuning for group '{ft_group}' with exclude_module='{exclude_module}'"
        )
        model = pick_trainable_params(
            model, ft_group=ft_group, exclude_module=exclude_module
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
            "gradient_clipping": 1.0,
            "steps_per_print": 2000,
            "wall_clock_breakdown": False,
        }

    deepspeed_cfg = (
        build_ds_config(args, args.zero_stage) if args.zero_stage is not None else None
    )

    privacy_args = dp_transformers.PrivacyArguments

    privacy_args.target_epsilon = args.target_epsilon
    privacy_args.target_delta = float(1 / len(train_ds))
    privacy_args.per_sample_max_grad_norm = args.per_sample_max_grad_norm
    privacy_args.kto_z_clip = args.kto_z_clip
    privacy_args.kto_z_noise_multiplier = args.kto_z_noise_multiplier

    training_method = args.method.lower()
    def _attach_epoch_logger(trainer_in):
        if trainer_in is not None and args.log_metrics_each_epoch:
            trainer_in.add_callback(EpochMetricsLoggerCallback())
        return trainer_in

    if training_method == "sft":
        trainer = build_sft_trainer(
            model_name=model,
            tokenizer=tokenizer,
            train_in=train_ds,
            eval_in=eval_ds,
            do_eval=args.do_eval,
            max_length=args.max_seq_len,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            lr=args.lr,
            bf16=True if args.bf16 else False,
            fp16=True if args.fp16 else False,
            logging_tool="wandb" if args.log_to_wandb else "none",
            loss_type=args.sft_loss_type,
            peft_config=peft_config,
            ds_config=deepspeed_cfg,
            privacy_args=privacy_args,
            ghost_mode=args.ghost_mode,
            per_sample_max_grad_norm=args.per_sample_max_grad_norm,
        )
        _attach_epoch_logger(trainer)
        train_start = time.perf_counter()
        trainer.train()
        log_phase_duration("trainer.train() [sft]", time.perf_counter() - train_start)

    elif training_method == "dpo":
        trainer = build_dpo_trainer(
            model_name=model,
            tokenizer=tokenizer,
            train_in=train_ds,
            eval_in=eval_ds,
            do_eval=args.do_eval,
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
            peft_config=peft_config,
            ds_config=deepspeed_cfg,
            privacy_args=privacy_args,
            ghost_mode=args.ghost_mode,
            per_sample_max_grad_norm=args.per_sample_max_grad_norm,
        )
        _attach_epoch_logger(trainer)
        train_start = time.perf_counter()
        trainer.train()
        log_phase_duration("trainer.train() [dpo]", time.perf_counter() - train_start)

    elif training_method == "kto":
        trainer = build_kto_trainer(
            model_name=model,
            tokenizer=tokenizer,
            train_in=train_ds,
            eval_in=eval_ds,
            do_eval=args.do_eval,
            max_length=args.max_seq_len,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            lr=args.lr,
            bf16=True if args.bf16 else False,
            fp16=True if args.fp16 else False,
            logging_tool="wandb" if args.log_to_wandb else "none",
            beta=args.kto_beta,
            peft_config=peft_config,
            ds_config=deepspeed_cfg,
            privacy_args=privacy_args,
            ghost_mode=args.ghost_mode,
            per_sample_max_grad_norm=args.per_sample_max_grad_norm,
        )
        _attach_epoch_logger(trainer)
        train_start = time.perf_counter()
        trainer.train()
        log_phase_duration("trainer.train() [kto]", time.perf_counter() - train_start)

    elif training_method == "orpo":
        trainer = build_orpo_trainer(
            model_name=model,
            tokenizer=tokenizer,
            train_in=train_ds,
            eval_in=eval_ds,
            do_eval=args.do_eval,
            max_length=args.max_seq_len,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            lr=args.lr,
            bf16=True if args.bf16 else False,
            fp16=True if args.fp16 else False,
            logging_tool="wandb" if args.log_to_wandb else "none",
            beta=args.orpo_beta,
            peft_config=peft_config,
            ds_config=deepspeed_cfg,
            privacy_args=privacy_args,
            ghost_mode=args.ghost_mode,
            per_sample_max_grad_norm=args.per_sample_max_grad_norm,
        )
        _attach_epoch_logger(trainer)
        train_start = time.perf_counter()
        trainer.train()
        log_phase_duration("trainer.train() [orpo]", time.perf_counter() - train_start)

    else:
        raise ValueError(f"Unknown training method: {training_method}")

    print(f"[*] Training Complete for Shadow {args.shadow_id}.")
    if trainer is not None:
        model = trainer.model
    trainer_log_history = []
    epoch_metrics = []
    if trainer is not None and args.log_metrics_each_epoch:
        trainer_log_history = list(getattr(trainer.state, "log_history", []) or [])
        epoch_metrics = summarize_epoch_metrics(trainer_log_history)
        print_epoch_metric_summary(epoch_metrics)

    model.eval()

    schemes = [
        SamplingScheme(name="greedy", type="top_k", k=1, T=1.0),
        SamplingScheme(name="topk40_t1", type="top_k", k=40, T=1.0),  # Paper's choice
        SamplingScheme(name="topp0.9_t1", type="top_p", p=0.9, T=1.0),
    ]

    def _evaluate_dataset(
        dataset_in,
        membership_mask_in,
        *,
        log_prefix: str,
        metrics_phase_name: str,
        summary_label: str,
        log_to_wandb_in: bool,
        allow_single_class_mia: bool = False,
    ) -> Optional[Dict[str, Any]]:
        metrics_start = time.perf_counter()
        loss_records = compute_metrics(
            ds=dataset_in,
            max_length=args.max_seq_len,
            model=model,
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
                f"{log_prefix} Rank {rank}/{world_size} skipping metrics aggregation; waiting for rank 0.",
                flush=True,
            )
            if dist is not None and dist.is_available() and dist.is_initialized():
                dist.barrier()
            return None

        utility_metrics, mia_scores = _compute_eval_summaries(loss_records)

        if log_to_wandb_in:
            wandb.log(utility_metrics)

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
                wandb.log({f"mia_{k}": v for k, v in mia_scores.items()})

        return {
            "utility": utility_metrics,
            "loss_records": loss_records,
            "mia_scores": mia_scores,
        }

    eval_artifacts = _evaluate_dataset(
        ds,
        membership_mask,
        log_prefix=f"[*] Shadow {args.shadow_id}",
        metrics_phase_name="compute_metrics()",
        summary_label="Utility (standard shadow universe)",
        log_to_wandb_in=args.log_to_wandb,
    )
    if eval_artifacts is None:
        return

    out = {
        "args": vars(args),
        "utility": eval_artifacts["utility"],
        "membership_mask": membership_mask,
        "loss_records": eval_artifacts["loss_records"],
        "mia_scores": eval_artifacts["mia_scores"],
        "special_canaries": special_canary_meta,
    }
    if args.log_metrics_each_epoch:
        out["epoch_metrics"] = epoch_metrics
        out["trainer_log_history"] = trainer_log_history

    if args.subdir is not None:
        if args.use_lora:
            save_path = os.path.join(
                args.output_dir,
                f"{args.dataset_name.split('/')[-1]}",
                f"{args.method}",
                "lora",
                f"{args.model_name.split('/')[-1]}",
                f"{args.subdir}",
            )
        elif args.enable_selective_finetuning:
            save_path = os.path.join(
                args.output_dir,
                f"{args.dataset_name.split('/')[-1]}",
                f"{args.method}",
                f"selective_ft_{args.ft_group}_exclude_{args.exclude_module}",
                f"{args.model_name.split('/')[-1]}",
                f"{args.subdir}",
            )
        else:
            save_path = os.path.join(
                args.output_dir,
                f"{args.dataset_name.split('/')[-1]}",
                f"{args.method}",
                f"{args.model_name.split('/')[-1]}",
                f"{args.subdir}",
            )
    else:
        if args.use_lora:
            save_path = os.path.join(
                args.output_dir,
                f"{args.dataset_name.split('/')[-1]}",
                f"{args.method}",
                "lora",
                f"{args.model_name.split('/')[-1]}",
            )
        elif args.enable_selective_finetuning:
            save_path = os.path.join(
                args.output_dir,
                f"{args.dataset_name.split('/')[-1]}",
                f"{args.method}",
                f"selective_ft_{args.ft_group}_exclude_{args.exclude_module}",
                f"{args.model_name.split('/')[-1]}",
            )
        else:
            save_path = os.path.join(
                args.output_dir,
                f"{args.dataset_name.split('/')[-1]}",
                f"{args.method}",
                f"{args.model_name.split('/')[-1]}",
            )
    os.makedirs(save_path, exist_ok=True)

    if training_method != "none":
        path_name = f"shadow_{args.shadow_id}.pkl"
    else:
        path_name = f"shadow_ref.pkl"

    save_start = time.perf_counter()
    with open(os.path.join(save_path, path_name), "wb") as f:
        pickle.dump(out, f)
    log_phase_duration("result serialization", time.perf_counter() - save_start)

    print(f"\n[*] Shadow {args.shadow_id}, Results saved to {save_path}.")
    log_phase_duration("total runtime", time.perf_counter() - total_start)
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
