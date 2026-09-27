from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, DPOConfig

try:
    from trl.experimental.kto import KTOConfig

    print("Imported KTOConfig from trl.experimental.kto")
except:
    from trl import KTOConfig

    print("Imported KTOConfig from trl")

try:
    from trl.experimental.orpo import ORPOConfig

    print("Imported ORPOConfig from trl.experimental.orpo")
except:
    from trl import ORPOConfig

    print("Imported ORPOConfig from trl")

from peft import LoraConfig
from src.train_utils_dp import (
    PrivateSFTTrainer,
    PrivateDPOTrainer,
    PrivateKTOTrainer,
    PrivateORPOTrainer,
)


def _attach_private_runtime_options(
    cfg,
    ghost_mode: bool = False,
    per_sample_max_grad_norm: float = 1.0,
):
    """
    Attach runtime-only DP flags expected by PrivateTrainer but not native to TRL configs.
    """
    setattr(cfg, "ghost_mode", bool(ghost_mode))
    setattr(cfg, "per_sample_max_grad_norm", float(per_sample_max_grad_norm))
    return cfg


def create_ref_model(model_name, ds_config: str = None):
    ref_model = None
    if ds_config is not None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name.config._name_or_path,
            dtype=getattr(model_name, "dtype", None),
            trust_remote_code=True,
        )
        ref_model.to("cuda")
    return ref_model


def build_sft_trainer(
    model_name,
    tokenizer,
    privacy_args,
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
    peft_config: LoraConfig = None,
    ds_config: str = None,
    ghost_mode: bool = False,
    per_sample_max_grad_norm: float = 1.0,
) -> PrivateSFTTrainer:
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
        gradient_checkpointing=False,
        loss_type=loss_type,
        deepspeed=ds_config,
    )
    cfg = _attach_private_runtime_options(
        cfg,
        ghost_mode=ghost_mode,
        per_sample_max_grad_norm=per_sample_max_grad_norm,
    )
    return PrivateSFTTrainer(
        model=model_name,
        args=cfg,
        privacy_args=privacy_args,
        train_dataset=curr_train,
        eval_dataset=curr_eval if do_eval == True else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def build_dpo_trainer(
    model_name,
    tokenizer,
    privacy_args,
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
    peft_config: LoraConfig = None,
    ds_config: str = None,
    ghost_mode: bool = False,
    per_sample_max_grad_norm: float = 1.0,
) -> PrivateDPOTrainer:
    mpl, mcl = int(round(max_length / 2)), int(round(max_length / 2))
    cfg = DPOConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        logging_steps=10,
        max_prompt_length=mpl,
        max_completion_length=mcl,
        max_length=max_length,
        save_strategy="no",
        do_eval=do_eval,
        eval_strategy="epoch" if do_eval == True else "no",
        bf16=bf16,
        fp16=fp16,
        report_to=logging_tool,
        loss_type=loss_type,
        beta=beta,
        gradient_checkpointing=False,
        deepspeed=ds_config,
    )
    cfg = _attach_private_runtime_options(
        cfg,
        ghost_mode=ghost_mode,
        per_sample_max_grad_norm=per_sample_max_grad_norm,
    )
    ref_model = create_ref_model(model_name, ds_config=ds_config)
    return PrivateDPOTrainer(
        model=model_name,
        ref_model=ref_model,
        args=cfg,
        privacy_args=privacy_args,
        processing_class=tokenizer,
        train_dataset=train_in,  # expects prompt/chosen/rejected
        eval_dataset=eval_in if do_eval == True else None,
        peft_config=peft_config,
    )


def build_kto_trainer(
    model_name,
    tokenizer,
    privacy_args,
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
    peft_config: LoraConfig = None,
    ds_config: str = None,
    ghost_mode: bool = False,
    per_sample_max_grad_norm: float = 1.0,
) -> PrivateKTOTrainer:
    mpl, mcl = int(round(max_length / 2)), int(round(max_length / 2))
    cfg = DPOConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        logging_steps=10,
        max_prompt_length=mpl,
        max_completion_length=mcl,
        max_length=max_length,
        save_strategy="no",
        bf16=bf16,
        fp16=fp16,
        report_to=logging_tool,
        do_eval=do_eval,
        eval_strategy="epoch" if do_eval == True else "no",
        loss_type="sigmoid",
        beta=beta,
        gradient_checkpointing=False,
        precompute_ref_log_probs=True,
        deepspeed=ds_config,
    )
    cfg.desirable_weight = 1.0
    cfg.undesirable_weight = 1.0
    cfg = _attach_private_runtime_options(
        cfg,
        ghost_mode=ghost_mode,
        per_sample_max_grad_norm=per_sample_max_grad_norm,
    )
    ref_model = create_ref_model(model_name, ds_config=ds_config)
    return PrivateKTOTrainer(
        model=model_name,
        ref_model=ref_model,
        args=cfg,
        privacy_args=privacy_args,
        processing_class=tokenizer,
        train_dataset=train_in,
        eval_dataset=eval_in if do_eval == True else None,
        peft_config=peft_config,
    )


def build_orpo_trainer(
    model_name,
    tokenizer,
    privacy_args,
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
    peft_config: LoraConfig = None,
    ds_config: str = None,
    ghost_mode: bool = False,
    per_sample_max_grad_norm: float = 1.0,
) -> PrivateORPOTrainer:
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
        gradient_checkpointing=False,
        deepspeed=ds_config,
    )
    cfg = _attach_private_runtime_options(
        cfg,
        ghost_mode=ghost_mode,
        per_sample_max_grad_norm=per_sample_max_grad_norm,
    )
    return PrivateORPOTrainer(
        model=model_name,
        args=cfg,
        privacy_args=privacy_args,
        processing_class=tokenizer,
        train_dataset=train_in,
        eval_dataset=eval_in if do_eval == True else None,
        peft_config=peft_config,
    )
