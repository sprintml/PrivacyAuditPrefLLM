# Benchmarking Membership Privacy Risks in Preference-Based LLM Post-Training

## Abstract

Modern language models are commonly adapted after pretraining to follow instructions, align with user preferences, and improve deployment behavior. Such post-training often relies on preference data from users, annotators, or model interactions, which may contain sensitive prompts, private responses, proprietary tasks, or confidential judgments. Understanding the privacy implications of this data is therefore essential. Membership inference attacks (MIAs), which test whether a record was used for training, are the de facto standard for empirical privacy auditing in machine learning. However, preference-based post-training changes the unit of membership: a training record contains a prompt, a preferred response, and a dispreferred response, rather than a single input-output sequence. Audits that ignore this structure can under-report leakage. We therefore introduce a benchmark protocol that adapts strong reference-model MIAs to complete preference records. Our protocol asks whether membership is exposed by either response alone, by the model’s preference between them, or by the two responses jointly. We find that auditing the response pair can reveal membership evidence missed when the record is reduced to a single score. Across three preference datasets, two model families, and seven post-training objectives spanning three post-training families, measured leakage depends strongly on both the audited record statistic and the training objective, with imbalanced risk between preferred and dispreferred responses. We further evaluate the impact of parameter-efficient fine-tuning (PEFT) and differential privacy (DP), finding widely varying privacy-utility trade-offs. Overall, preference-based post-training leaks in ways that standard single-response audits can miss, motivating formal privacy methods that protect complete preference records while preserving post-training utility.

## Setup

This repository trains shadow models for preference-optimization experiments,
scores membership inference attacks (MIAs), and saves attack result pickles from
the resulting shadow-model outputs.

The main workflow is:

1. Train one shadow model per `shadow_id`.
2. Score the saved `shadow_*.pkl` files with LiRA or RMIA.

## Repository Layout

- `scripts/train_shadow.py`: non-private shadow training for SFT, DPO, KTO, ORPO, reward models, PPO, GRPO, and adapter variants.
- `scripts/train_shadow_dp.py`: DP shadow training for SFT, DPO, KTO, and ORPO.
- `scripts/run_mia.py`: attack scoring over directories containing `shadow_*.pkl`.
- `scripts/print_canary_examples.py`: inspect generated canary examples.
- `src/`: shared training, feature extraction, attack, and utility code.
- `environment.yaml`: pinned conda environment.

Generated outputs are intentionally ignored by git. By default, examples below
write under `out/`.

## Installation

### 1. Create the environment

```bash
conda env create -f environment.yaml
conda activate rlpa
```

The environment uses Python 3.11 and includes PyTorch, Transformers, TRL,
datasets, PEFT, pandas, scipy, and W&B.

DP training imports `opacus` and `dp_transformers`. If those packages are not
available in your environment, install them before using
`scripts/train_shadow_dp.py`:

```bash
python3 -m pip install opacus dp-transformers
```

### 2. Verify the checkout

```bash
python3 -m compileall -q src scripts
python scripts/run_mia.py --help
python scripts/train_shadow.py --help
```

### 3. Configure optional external services

If you use gated Hugging Face models or datasets, authenticate before running
training:

```bash
huggingface-cli login
```

If you enable W&B logging with `--log_to_wandb`, set the project name through an
environment variable:

```bash
export WANDB_PROJECT=my_project_name
```

No machine-specific paths are required. All examples use relative paths.

## Supported Data Format

The trainer expects a preference dataset with:

- `prompt`
- `chosen`
- `rejected`

The helper loader has built-in mappings for:

- `HuggingFaceH4/ultrafeedback_binarized`
- `trl-lib/hh-rlhf-helpful-base`
- `Intel/orca_dpo_pairs`
- `chatbot_arena_2024` (loaded from the corresponding split of the public preference dataset used by the code)

Other datasets can work if their loaded split can be mapped to the same
`prompt/chosen/rejected` structure.

## End-to-End Quickstart

The commands below run a small, local four-shadow experiment. Increase
`--num_shadows`, `--train_samples`, epochs, model size, and batch settings for
real experiments.

### 1. Train Shadow Models

Each `shadow_id` writes one `shadow_<id>.pkl` into a shared run directory. The
directory structure is:

```text
<output_dir>/<dataset_slug>/<method>/<model_slug>[/<subdir>]/
```

Example SFT run:

```bash
for sid in 0 1 2 3; do
    python scripts/train_shadow.py \
        --method sft \
        --dataset_name HuggingFaceH4/ultrafeedback_binarized \
        --model_name Qwen/Qwen3-0.6B \
        --num_shadows 4 \
        --shadow_id "${sid}" \
        --shadow_seed 123 \
        --train_samples 2000 \
        --max_seq_len 256 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 1 \
        --lr 2e-5 \
        --bf16 \
        --output_dir out/models
done
```

For the example above, the attack input directory is:

```bash
export RUN_DIR="out/models/ultrafeedback_binarized/sft/Qwen3-0.6B"
```

Supported non-private methods are:

```text
sft, dpo, ppo, grpo, kto, orpo, reward, none
```

Useful training flags:

- `--subdir <name>`: add a final path component so related runs do not collide.
- `--use_peft --peft_version lora`: train adapter variants.
- `--save_epoch_snapshots`: save intermediate `shadow_*.pkl` files under epoch subdirectories.
- `--log_metrics_each_epoch`: print and persist epoch-level metrics.
- `--log_to_wandb`: enable W&B logging; use `WANDB_PROJECT` for the project name.

### 2. Train DP Shadow Models

DP runs use `scripts/train_shadow_dp.py` and write to `out/models_dp` by
default. Supported methods are:

```text
sft, dpo, kto, orpo
```

Minimal DP-KTO example:

```bash
python scripts/train_shadow_dp.py \
    --method kto \
    --dataset_name chatbot_arena_2024 \
    --model_name Qwen/Qwen3-0.6B \
    --num_shadows 4 \
    --shadow_id 0 \
    --shadow_seed 123 \
    --train_samples 2000 \
    --max_seq_len 256 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --lr 1e-4 \
    --bf16 \
    --target_epsilon 10.0 \
    --per_sample_max_grad_norm 1e-2 \
    --kto_beta 0.1 \
    --kto_z_clip 1.0 \
    --output_dir out/models_dp
```

DP-KTO uses a cyclic shifted reference-point release for the KTO `z` term and
tracks it jointly with the DP-SGD step. If `--kto_z_noise_multiplier` is not
set, the trainer calibrates the main noise to the requested target epsilon.

### 3. Add Canary Samples

Both training scripts accept canary counters. Canary samples are appended before
the shadow split, and metadata is saved in each shadow pickle under
`special_canaries`.

Example:

```bash
python scripts/train_shadow.py \
    --method dpo \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --model_name Qwen/Qwen3-0.6B \
    --num_shadows 4 \
    --shadow_id 0 \
    --train_samples 2000 \
    --num_normal_canaries 10 \
    --num_mislabeled_canaries 10 \
    --bf16
```

Available canary counters:

- `--num_normal_canaries`
- `--num_random_string_canaries`
- `--num_mislabeled_canaries`
- `--num_random_question_canaries`
- `--num_reasonable_question_canaries`
- `--num_reasonable_question_random_chosen_canaries`

## Scoring Membership Inference Attacks

Run `scripts/run_mia.py` on a directory containing the complete set of
`shadow_*.pkl` files:

```bash
python scripts/run_mia.py \
    --path_to_pickle "${RUN_DIR}" \
    --attack_method lira \
    --feature raw_logprob \
    --stream chosen \
    --truncation_strategy tail \
    --fixed_d 64
```

The script writes a result file into the same run directory:

```text
new_results_<feature>_<stream>_<attack>_D<fixed_d>_<timestamp>.pkl
```

### Common Attack Commands

LiRA:

```bash
python scripts/run_mia.py \
    --path_to_pickle "${RUN_DIR}" \
    --attack_method lira \
    --feature raw_logprob \
    --stream chosen \
    --lira_covariance_estimator univariate \
    --fixed_d 64
```

RMIA:

```bash
python scripts/run_mia.py \
    --path_to_pickle "${RUN_DIR}" \
    --attack_method rmia \
    --feature raw_logprob \
    --stream both \
    --truncation_strategy half_pad_concat \
    --rmia_mode multivariate_exact \
    --rmia_version standard \
    --rmia_log_gamma 0.0 \
    --fixed_d 128
```

### Batch Attack Settings

Use `--settings_file` to run several attacks after loading shadows once. The
file can be JSON or JSONL.

Example `attack_settings.json`:

```json
[
  {
    "attack_method": "lira",
    "feature": "raw_logprob",
    "stream": "chosen",
    "fixed_d": 64,
    "lira_covariance_estimator": "univariate"
  },
  {
    "attack_method": "rmia",
    "feature": "raw_logprob",
    "stream": "both",
    "truncation_strategy": "half_pad_concat",
    "fixed_d": 128,
    "rmia_mode": "multivariate_exact",
    "rmia_version": "standard",
    "rmia_log_gamma": 0.0
  }
]
```

Run:

```bash
python scripts/run_mia.py \
    --path_to_pickle "${RUN_DIR}" \
    --settings_file attack_settings.json
```

### Important Attack Flags

- `--attack_method`: `lira` or `rmia`.
- `--feature`: token or scalar feature to attack. Common choices are
  `raw_logprob`, `min_kpp`, `hinge`, `stable`, `logit`, `dpo`, `ipo`,
  `premia`, `kto`, and `orpo`.
- `--stream`: `chosen`, `rejected`, or `both`.
- `--truncation_strategy`: `tail`, `average`, `half_pad_concat`, or grouped
  reducers such as `average_8`, `min_8`, `max_8`.
- `--fixed_d`: explicit vector length for token features. If omitted, the code
  derives `D` from `--k_pct`.
- `--num_in_models` and `--num_out_models`: optional controls for how many IN
  and OUT shadow models are used per scored sample.
- `--torch_threads` and `--torch_interop_threads`: reduce CPU oversubscription
  when running many jobs.

For token features, variable-length token sequences are converted to fixed-size
vectors before scoring. For scalar features such as DPO/IPO/KTO/ORPO-style
losses, `D` and token truncation are ignored.

## Utility Scripts

Print example canary rows:

```bash
python scripts/print_canary_examples.py \
    --datasets HuggingFaceH4/ultrafeedback_binarized \
    --per-type 2
```

## Path and Environment Configuration

The code is designed to avoid hardcoded user or machine paths. Useful variables:

| Variable        | Used by          | Purpose                                            |
| --------------- | ---------------- | -------------------------------------------------- |
| `WANDB_PROJECT` | training scripts | W&B project name when `--log_to_wandb` is enabled. |

## Output Files

Training outputs:

- `shadow_<id>.pkl`: one per shadow model.
- `shadow_ref.pkl`: reference output for `--method none`.
- `metrics.pkl`: trainer metrics.
- `epoch_metrics.pkl`: optional epoch metrics.
- model/tokenizer files from `save_pretrained` when applicable.

Attack outputs:

- `new_results_*.pkl`: saved by `scripts/run_mia.py` in the run directory.

## Troubleshooting

- `ModuleNotFoundError`: activate the conda environment with `conda activate rlpa`.
- Dataset download fails: authenticate with Hugging Face if needed and make sure
  the dataset name is accessible.
- CUDA out of memory: reduce `--train_samples`, `--max_seq_len`,
  `--per_device_train_batch_size`, or model size; increase
  `--gradient_accumulation_steps`; or enable adapter training.
- `run_mia.py` finds too few samples: check that all expected `shadow_*.pkl`
  files are in one directory and that they were trained with the same
  `--num_shadows` and dataset size.
