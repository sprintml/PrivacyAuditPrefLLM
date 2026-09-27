import os
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import (
    SFTTrainer,
    SFTConfig,
    DPOTrainer,
    DPOConfig,
    GRPOTrainer,
    GRPOConfig,
    RewardTrainer,
    RewardConfig,
    ORPOTrainer,
    ORPOConfig,
)

try:
    from trl.experimental.orpo import ORPOTrainer, ORPOConfig

    print("Imported ORPOTrainer and ORPOConfig from trl.experimental.orpo")
except:
    from trl import ORPOTrainer, ORPOConfig

    print("Imported ORPOTrainer and ORPOConfig from trl")
try:
    from trl.experimental.kto import KTOTrainer, KTOConfig

    print("Imported KTOTRainer, KTOConfig from trl.experimental.kto")
except:
    from trl import KTOTrainer, KTOConfig

    print("Imported KTOTrainer and KTOConfig from trl")
try:
    from trl.experimental.ppo import PPOTrainer, PPOConfig

    print("Imported PPOTrainer, PPOConfig from trl.experimental.ppo")
except:
    from trl import PPOTrainer, PPOConfig

    print("Imported PPOTrainer and PPOConfig from trl")


def create_ref_model(model_or_name, ds_config: str = None):
    ref_model = None
    if ds_config is not None:
        if isinstance(model_or_name, str):
            ref_name = model_or_name
            ref_dtype = None
        else:
            ref_name = model_or_name.config._name_or_path
            ref_dtype = getattr(model_or_name, "dtype", None)
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_name,
            dtype=ref_dtype,
            trust_remote_code=True,
        )
        ref_model.to("cuda")
    return ref_model


def build_reward_trainer(
    model_name,
    tokenizer,
    train_in: Dataset,
    eval_in: Dataset,
    do_eval: bool,
    max_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    lr: float,
    bf16: bool = True,
    fp16: bool = False,
    logging_tool: str = "none",
    peft_config=None,
    ds_config=None,
) -> RewardTrainer:
    # mpl, mcl = int(round(max_length / 2)), int(round(max_length / 2))
    cfg = RewardConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        max_length=max_length,
        logging_steps=10,
        save_strategy="no",
        bf16=bf16,
        fp16=fp16,
        report_to=logging_tool,
        do_eval=do_eval,
        eval_strategy="epoch" if do_eval == True else "no",
        gradient_checkpointing=True,
        # use_liger_kernel=True,
    )
    if isinstance(model_name, str):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            trust_remote_code=True,
        )
    else:
        base_model = model_name
    trainer = RewardTrainer(
        model=base_model,
        processing_class=tokenizer,
        train_dataset=train_in,
        eval_dataset=eval_in if do_eval == True else None,
        args=cfg,
        peft_config=peft_config,
    )
    return trainer


def build_sft_trainer(
    model_name,
    tokenizer,
    train_in: Dataset,
    eval_in: Dataset,
    do_eval: bool,
    max_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    lr: float,
    bf16: bool = True,
    fp16: bool = False,
    logging_tool: str = "none",
    loss_type: str = "nll",
    peft_config=None,
    ds_config: str = None,
) -> SFTTrainer:
    # Pre-tokenize with explicit masking for robustness across model templates
    curr_train = train_in.map(
        lambda ex: {"messages": ex["chosen"]},
        remove_columns=train_in.column_names,
        batched=True,
    )
    curr_eval = eval_in.map(
        lambda ex: {"messages": ex["chosen"]},
        remove_columns=eval_in.column_names,
        batched=True,
    )
    cfg = SFTConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        logging_steps=10,
        max_length=max_length,
        save_strategy="no",
        bf16=bf16,
        fp16=fp16,
        report_to=logging_tool,
        do_eval=do_eval,
        eval_strategy="epoch" if do_eval == True else "no",
        gradient_checkpointing=True,
        loss_type=loss_type,
    )
    return SFTTrainer(
        model=model_name,
        args=cfg,
        train_dataset=curr_train,
        eval_dataset=curr_eval if do_eval == True else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def build_dpo_trainer(
    model_or_name,
    tokenizer,
    train_in: Dataset,
    eval_in: Dataset,
    do_eval: bool,
    max_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    lr: float,
    bf16: bool = True,
    fp16: bool = False,
    logging_tool: str = "none",
    loss_type: str = "sigmoid",
    beta: float = 0.1,
    peft_config=None,
    optim: str = None,
    ds_config: str = None,
) -> DPOTrainer:
    cfg_kwargs = {
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "learning_rate": lr,
        "logging_steps": 10,
        "max_length": max_length,
        "save_strategy": "no",
        "do_eval": do_eval,
        "eval_strategy": "epoch" if do_eval == True else "no",
        "bf16": bf16,
        "fp16": fp16,
        "report_to": logging_tool,
        "loss_type": loss_type,
        "beta": beta,
        # use_liger_kernel=True,
        "gradient_checkpointing": True,
    }
    if optim is not None:
        cfg_kwargs["optim"] = optim
    cfg = DPOConfig(**cfg_kwargs)
    if isinstance(model_or_name, str):
        model = AutoModelForCausalLM.from_pretrained(model_or_name)
    else:
        model = model_or_name
    ref_model = create_ref_model(model_or_name, ds_config=ds_config)
    return DPOTrainer(
        model=model,
        ref_model=ref_model if peft_config is None else None,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=train_in,  # expects prompt/chosen/rejected
        eval_dataset=eval_in if do_eval == True else None,
        peft_config=peft_config,
    )


def build_ppo_trainer(
    model_name,
    reward_model_id,
    tokenizer,
    train_in: Dataset,
    eval_in: Dataset,
    do_eval: bool,
    max_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    lr: float,
    bf16: bool = True,
    fp16: bool = False,
    logging_tool: str = "none",
    peft_config=None,
    ds_config: str = None,
) -> PPOTrainer:
    print("\n[*] Training Reward Model for GRPO\n")
    mpl, mcl = int(round(max_length / 2)), int(round(max_length / 2))

    if isinstance(reward_model_id, str):
        reward_model = build_reward_trainer(
            model_name=reward_model_id,
            tokenizer=tokenizer,
            train_in=train_in,
            eval_in=eval_in,
            do_eval=do_eval,
            max_length=max_length,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr=lr,
            bf16=bf16,
            fp16=fp16,
            logging_tool=logging_tool,
            peft_config=peft_config,
            ds_config=None,
        )
        reward_model.train()
    else:
        reward_model = None

    print("\n[*] Reward Model Training Complete. Starting PPO Training.\n")

    cfg = PPOConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        response_length=mcl,
        logging_steps=10,
        save_strategy="no",
        do_eval=do_eval,
        eval_strategy="epoch" if do_eval == True else "no",
        num_sample_generations=0,
        bf16=bf16,
        fp16=fp16,
        report_to=logging_tool,
        gradient_checkpointing=True,
        local_rollout_forward_batch_size=2,
        torch_empty_cache_steps=1,
        max_grad_norm=0.1,
        deepspeed=ds_config,
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )

    def _tokenize_ppo_row(example):
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(
            prompt_text,
            truncation=True,
            max_length=mpl,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    ppo_train = train_in.map(
        _tokenize_ppo_row,
        remove_columns=train_in.column_names,
    )

    ppo_eval = (
        eval_in.map(
            _tokenize_ppo_row,
            remove_columns=eval_in.column_names,
        )
        if do_eval
        else None
    )

    return PPOTrainer(
        model=model_name,
        ref_model=(
            AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
            if ds_config is not None
            else None
        ),
        reward_model=(
            reward_model.model if reward_model is not None else reward_model_id
        ),
        value_model=AutoModelForSequenceClassification.from_pretrained(
            reward_model_id,
            num_labels=1,
        ),
        processing_class=tokenizer,
        args=cfg,
        train_dataset=ppo_train,
        eval_dataset=ppo_eval,
        peft_config=peft_config,
    )


def build_grpo_trainer(
    model_name,
    reward_model_id,
    tokenizer,
    train_in: Dataset,
    eval_in: Dataset,
    do_eval: bool,
    max_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    num_train_epochs: int,
    lr: float,
    bf16: bool = True,
    fp16: bool = False,
    logging_tool: str = "none",
    loss_type: str = "dapo",
    beta: float = 0.0,
    epsilon: float = 0.2,
    importance_sampling_level: str = "token",
    peft_config=None,
    ds_config: str = None,
) -> GRPOTrainer:
    print("\n[*] Training Reward Model for GRPO\n")
    mpl, mcl = int(round(max_length / 2)), int(round(max_length / 2))

    if isinstance(reward_model_id, str):
        reward_model = build_reward_trainer(
            model_name=reward_model_id,
            tokenizer=tokenizer,
            train_in=train_in,
            eval_in=eval_in,
            do_eval=do_eval,
            max_length=max_length,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            lr=lr,
            bf16=bf16,
            fp16=fp16,
            logging_tool=logging_tool,
            peft_config=peft_config,
            ds_config=None,
        )
        reward_model.train()
    else:
        reward_model = None

    print("\n[*] Reward Model Training Complete. Starting GRPO Training.\n")

    per_device_eval_batch_size = per_device_train_batch_size
    if per_device_eval_batch_size % num_generations != 0:
        new_eval_bs = (
            (per_device_eval_batch_size + num_generations - 1) // num_generations
        ) * num_generations
        print(
            f"[!] Updating per_device_eval_batch_size from "
            f"{per_device_eval_batch_size} to {new_eval_bs} so it is divisible "
            f"by num_generations={num_generations}"
        )
        per_device_eval_batch_size = new_eval_bs

    cfg = GRPOConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        logging_steps=10,
        max_prompt_length=mpl,
        max_completion_length=mcl,
        save_strategy="no",
        num_generations=num_generations,
        do_eval=do_eval,
        eval_strategy="epoch" if do_eval == True else "no",
        bf16=bf16,
        fp16=fp16,
        report_to=logging_tool,
        loss_type=loss_type,
        beta=beta,
        importance_sampling_level=importance_sampling_level,
        epsilon=epsilon,
        gradient_checkpointing=True,
    )
    return GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=(
            reward_model.model if reward_model is not None else reward_model_id
        ),
        reward_processing_classes=tokenizer,
        args=cfg,
        train_dataset=train_in.select_columns(["prompt"]),
        eval_dataset=eval_in.select_columns(["prompt"]) if do_eval == True else None,
        peft_config=peft_config,
    )


def build_kto_trainer(
    model_name,
    tokenizer,
    train_in: Dataset,
    eval_in: Dataset,
    do_eval: bool,
    max_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    lr: float,
    bf16: bool = True,
    fp16: bool = False,
    logging_tool: str = "none",
    beta: float = 0.1,
    peft_config=None,
    ds_config: str = None,
) -> KTOTrainer:
    cfg = KTOConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        logging_steps=10,
        max_length=max_length,
        save_strategy="no",
        bf16=bf16,
        fp16=fp16,
        report_to=logging_tool,
        do_eval=do_eval,
        eval_strategy="epoch" if do_eval == True else "no",
        beta=beta,
        use_liger_kernel=False,
        gradient_checkpointing=True,
        precompute_ref_log_probs=True,
    )
    ref_model = create_ref_model(model_name, ds_config=ds_config)
    return KTOTrainer(
        model=model_name,
        ref_model=ref_model,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=train_in,
        eval_dataset=eval_in if do_eval == True else None,
        peft_config=peft_config,
    )


def build_orpo_trainer(
    model_name,
    tokenizer,
    train_in: Dataset,
    eval_in: Dataset,
    do_eval: bool,
    max_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    lr: float,
    bf16: bool = True,
    fp16: bool = False,
    logging_tool: str = "none",
    beta: float = 0.1,
    peft_config=None,
    ds_config: str = None,
) -> ORPOTrainer:
    mpl, mcl = int(round(max_length / 4)), int(round(max_length / 2))
    cfg = ORPOConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        logging_steps=10,
        max_length=max_length,
        max_prompt_length=mpl,
        max_completion_length=mcl,
        save_strategy="no",
        bf16=bf16,
        fp16=fp16,
        report_to=logging_tool,
        do_eval=do_eval,
        eval_strategy="epoch" if do_eval == True else "no",
        beta=beta,
        gradient_checkpointing=True,
    )
    return ORPOTrainer(
        model=model_name,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=train_in,
        eval_dataset=eval_in if do_eval == True else None,
        peft_config=peft_config,
    )
