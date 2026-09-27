import inspect
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Union

import datasets
import opacus
import torch
from dp_transformers import arguments, dp_utils, sampler
from opacus.accountants import RDPAccountant
from prv_accountant import Accountant as PRVAccountant
from torch import nn
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainerControl, TrainerState, modeling_utils
from transformers import training_args
from transformers.file_utils import is_datasets_available, is_sagemaker_mp_enabled
from trl import DPOTrainer, SFTTrainer

try:
    from trl.experimental.orpo import ORPOTrainer

    print("Imported ORPOTrainer from trl.experimental.orpo")
except:
    from trl import ORPOTrainer

    print("Imported ORPOTrainer from trl")


class _AttributeForwardingMixin:
    """
    Mixin that forwards missing attributes/methods to the wrapped model.
    """

    def __getattr__(self, item: str):
        try:
            return super().__getattr__(item)
        except AttributeError as exc:
            wrapped_model = self.__dict__.get("_module", None)
            if wrapped_model is not None and hasattr(wrapped_model, item):
                return getattr(wrapped_model, item)
            raise exc

    @contextmanager
    def no_sync(self):
        yield


class _ForwardingGradSampleModule(_AttributeForwardingMixin, dp_utils.GradSampleModule):
    """
    Standard GradSampleModule variant with attribute forwarding.
    """


class _ForwardingFastGradSampleModule(
    _AttributeForwardingMixin, opacus.GradSampleModuleFastGradientClipping
):
    """
    Fast/Ghost clipping GradSampleModule variant with attribute forwarding.
    """


def _untie_shared_trainable_parameters(model: nn.Module) -> List[tuple[str, str]]:
    """
    Ghost clipping in Opacus doesn't support shared trainable parameters.
    Clone duplicate references so every trainable parameter is unique.
    """
    seen_params: Dict[int, str] = {}
    untied_pairs: List[tuple[str, str]] = []

    for module_name, module in model.named_modules():
        module_params = list(module.named_parameters(recurse=False))
        for param_name, param in module_params:
            if param is None or not param.requires_grad:
                continue

            param_id = id(param)
            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if param_id not in seen_params:
                seen_params[param_id] = full_name
                continue

            source_name = seen_params[param_id]
            cloned_param = nn.Parameter(
                param.detach().clone(), requires_grad=param.requires_grad
            )
            setattr(module, param_name, cloned_param)
            untied_pairs.append((full_name, source_name))

    if untied_pairs and hasattr(model, "config") and hasattr(
        model.config, "tie_word_embeddings"
    ):
        model.config.tie_word_embeddings = False

    return untied_pairs


class PrivacyBudgetCallback(TrainerCallback):
    """
    Logs privacy spend alongside training logs.
    """

    def __init__(
        self,
        rdp_accountant: RDPAccountant,
        prv_accountant: PRVAccountant,
        target_delta: float,
        target_epsilon: Optional[float] = None,
        extra_rdp_accountant: Optional[RDPAccountant] = None,
        extra_prv_accountant: Optional[PRVAccountant] = None,
        combined_rdp_accountant: Optional[RDPAccountant] = None,
        combined_prv_accountant: Optional[PRVAccountant] = None,
        combined_prv_step_multiplier: int = 1,
    ) -> None:
        self.rdp_accountant = rdp_accountant
        self.prv_accountant = prv_accountant
        self.target_delta = target_delta
        self.target_epsilon = target_epsilon
        self.extra_rdp_accountant = extra_rdp_accountant
        self.extra_prv_accountant = extra_prv_accountant
        self.combined_rdp_accountant = combined_rdp_accountant
        self.combined_prv_accountant = combined_prv_accountant
        self.combined_prv_step_multiplier = combined_prv_step_multiplier

    def on_log(
        self,
        args: training_args.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Dict[str, Any],
    ) -> TrainerControl:
        if logs is None:
            return control

        eps_rdp = None
        eps_prv = None
        eps_rdp_extra = None
        eps_prv_extra = None

        try:
            eps_rdp = float(self.rdp_accountant.get_epsilon(self.target_delta))
        except Exception:
            pass

        try:
            eps_prv = float(self.prv_accountant.compute_epsilon(state.global_step)[2])
        except Exception:
            pass

        if self.combined_rdp_accountant is not None:
            try:
                logs["privacy/epsilon_rdp"] = float(
                    self.combined_rdp_accountant.get_epsilon(self.target_delta)
                )
                if eps_rdp is not None:
                    logs["privacy/epsilon_rdp_main"] = eps_rdp
            except Exception:
                pass

        if self.combined_prv_accountant is not None:
            try:
                logs["privacy/epsilon_prv"] = float(
                    self.combined_prv_accountant.compute_epsilon(
                        self.combined_prv_step_multiplier * state.global_step
                    )[2]
                )
                if eps_prv is not None:
                    logs["privacy/epsilon_prv_main"] = eps_prv
            except Exception:
                pass

        if self.extra_rdp_accountant is not None:
            try:
                eps_rdp_extra = float(
                    self.extra_rdp_accountant.get_epsilon(self.target_delta)
                )
                logs["privacy/epsilon_rdp_aux"] = eps_rdp_extra
            except Exception:
                pass

        if self.extra_prv_accountant is not None:
            try:
                eps_prv_extra = float(
                    self.extra_prv_accountant.compute_epsilon(state.global_step)[2]
                )
                logs["privacy/epsilon_prv_aux"] = eps_prv_extra
            except Exception:
                pass

        if eps_rdp is not None and "privacy/epsilon_rdp" not in logs:
            logs["privacy/epsilon_rdp_main"] = eps_rdp
            logs["privacy/epsilon_rdp"] = eps_rdp + (
                eps_rdp_extra if eps_rdp_extra is not None else 0.0
            )

        if eps_prv is not None and "privacy/epsilon_prv" not in logs:
            logs["privacy/epsilon_prv_main"] = eps_prv
            logs["privacy/epsilon_prv"] = eps_prv + (
                eps_prv_extra if eps_prv_extra is not None else 0.0
            )

        logs["privacy/delta"] = float(self.target_delta)

        if self.target_epsilon is not None and self.target_epsilon > 0:
            logs["privacy/epsilon_target"] = float(self.target_epsilon)
            if "privacy/epsilon_rdp" in logs:
                logs["privacy/spent_frac_rdp"] = logs["privacy/epsilon_rdp"] / float(
                    self.target_epsilon
                )
            if "privacy/epsilon_prv" in logs:
                logs["privacy/spent_frac_prv"] = logs["privacy/epsilon_prv"] / float(
                    self.target_epsilon
                )

        return control


class AdditionalPrivacyBudgetCallback(TrainerCallback):
    """
    Steps an auxiliary RDP accountant once per optimizer step for an extra
    subsampled Gaussian mechanism composed with DP-SGD.
    """

    def __init__(
        self, accountant_steps: List[tuple[RDPAccountant, float, float, int]]
    ):
        self.accountant_steps = accountant_steps

    def on_step_end(
        self,
        args: training_args.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any],
    ) -> TrainerControl:
        try:
            for accountant, noise_multiplier, sample_rate, repetitions in self.accountant_steps:
                for _ in range(repetitions):
                    accountant.step(
                        noise_multiplier=noise_multiplier, sample_rate=sample_rate
                    )
        except Exception:
            pass
        return control


def _wrap_model_for_private_training(
    model: nn.Module, args: arguments.TrainingArguments
) -> nn.Module:
    """
    Wrap model once for DP training (DPDDP + GradSampleModule) without adding hooks twice.
    Selects fast ghost clipping wrapper only when args.ghost_mode is enabled.
    """
    private_model = model
    ghost_mode = bool(getattr(args, "ghost_mode", False))
    per_sample_max_grad_norm = float(getattr(args, "per_sample_max_grad_norm", 1.0))

    if isinstance(
        private_model, (_ForwardingGradSampleModule, _ForwardingFastGradSampleModule)
    ):
        if ghost_mode and isinstance(private_model, _ForwardingFastGradSampleModule):
            return private_model
        if not ghost_mode and isinstance(private_model, _ForwardingGradSampleModule):
            return private_model
        private_model = private_model.to_standard_module()

    if ghost_mode and not isinstance(private_model, opacus.GradSampleModule):
        untied_pairs = _untie_shared_trainable_parameters(private_model)
        if untied_pairs:
            preview = ", ".join(
                f"{dst} <- {src}" for dst, src in untied_pairs[:3]
            )
            if len(untied_pairs) > 3:
                preview = f"{preview}, ..."
            print(
                f"[DP] Ghost mode: untied {len(untied_pairs)} shared trainable "
                f"parameter reference(s) to satisfy Opacus ghost clipping "
                f"({preview})."
            )

    if isinstance(private_model, opacus.GradSampleModule):
        private_model = private_model.to_standard_module()

    if args.parallel_mode == training_args.ParallelMode.DISTRIBUTED and not isinstance(
        private_model, opacus.distributed.DifferentiallyPrivateDistributedDataParallel
    ):
        private_model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(
            private_model
        )

    if ghost_mode:
        return _ForwardingFastGradSampleModule(
            private_model,
            max_grad_norm=per_sample_max_grad_norm,
            use_ghost_clipping=True,
        )

    return _ForwardingGradSampleModule(private_model)


def _compose_private_callbacks(
    kwargs: Dict[str, Any], private_callbacks: List[TrainerCallback]
) -> List[TrainerCallback]:
    user_callbacks = kwargs.pop("callbacks", None)
    if user_callbacks is None:
        return private_callbacks
    if isinstance(user_callbacks, tuple):
        user_callbacks = list(user_callbacks)
    elif not isinstance(user_callbacks, list):
        user_callbacks = [user_callbacks]
    return [*private_callbacks, *user_callbacks]


class PrivateTrainer:
    """
    Shared DP logic for TRL trainers:
        (i) remove Trainer loss scaling by gradient_accumulation_steps in training_step
        (ii) use author-level sampler and dataloader
        (iii) wrap optimizer with Opacus DP optimizer
    """

    def __init__(
        self,
        model: Union[
            modeling_utils.PreTrainedModel, torch.nn.modules.module.Module
        ] = None,
        args: arguments.TrainingArguments = None,
        train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
        privacy_args: arguments.PrivacyArguments = None,
        author_mapping: Optional[Sequence[Sequence[int]]] = None,
        **kwargs: Dict,
    ) -> None:
        self.train_args = args
        self.privacy_args = privacy_args

        if author_mapping is None:
            if train_dataset is None:
                raise ValueError(
                    "PrivateTrainer requires `train_dataset` when `author_mapping` is not provided."
                )
            author_mapping = [[i] for i in range(len(train_dataset))]
        self.author_mapping = author_mapping

        if self.privacy_args.target_delta is None:
            self.privacy_args.target_delta = 1.0 / len(self.author_mapping)

        if self.privacy_args.noise_multiplier is None:
            self.privacy_args.noise_multiplier = self._find_noise_multiplier()

        self.rdp_accountant = RDPAccountant()
        self.prv_accountant = PRVAccountant(
            noise_multiplier=self.privacy_args.noise_multiplier,
            sampling_probability=self.sampling_probability,
            delta=self.privacy_args.target_delta,
            eps_error=0.1,
            max_compositions=self.num_steps,
        )

        self._init_extra_privacy_accounting()

        self.dp_callback = dp_utils.DPCallback(
            noise_multiplier=self.privacy_args.noise_multiplier,
            target_delta=self.privacy_args.target_delta,
            sampling_probability=self.sampling_probability,
            rdp_accountant=self.rdp_accountant,
            prv_accountant=self.prv_accountant,
        )
        self.privacy_budget_callback = self._build_privacy_budget_callback()
        callbacks = _compose_private_callbacks(
            kwargs,
            [
                self.dp_callback,
                self.privacy_budget_callback,
                *self._get_additional_private_callbacks(),
            ],
        )

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            callbacks=callbacks,
            **kwargs,
        )

        private_model = _wrap_model_for_private_training(self.model, args)
        self.model = private_model
        self.model_wrapped = private_model

        print("Privacy Arguments:")
        print(f"Noise Multiplier: {self.privacy_args.noise_multiplier}")
        print(f"Sampling Probability: {self.sampling_probability}")
        print(f"Target Delta: {self.privacy_args.target_delta}")
        if self.extra_prv_accountant is not None:
            print("Additional DP mechanism: enabled")
        print("Epsilon Error: 0.1")
        print(f"Max Compositions: {self.num_steps}")
        print("=" * 100)

        self.get_rdp_epsilon = lambda: (
            self.combined_rdp_accountant.get_epsilon(self.privacy_args.target_delta)
            if self.combined_rdp_accountant is not None
            else self.rdp_accountant.get_epsilon(self.privacy_args.target_delta)
            + (
                self.extra_rdp_accountant.get_epsilon(self.privacy_args.target_delta)
                if self.extra_rdp_accountant is not None
                else 0.0
            )
        )
        self.get_prv_epsilon = lambda: (
            self.combined_prv_accountant.compute_epsilon(
                self.combined_prv_step_multiplier * self.state.global_step
            )[2]
            if self.combined_prv_accountant is not None
            else self.prv_accountant.compute_epsilon(self.state.global_step)[2]
            + (
                self.extra_prv_accountant.compute_epsilon(self.state.global_step)[2]
                if self.extra_prv_accountant is not None
                else 0.0
            )
        )

    def _find_noise_multiplier(self) -> float:
        return arguments.find_noise_multiplier(
            target_epsilon=self.privacy_args.target_epsilon,
            target_delta=self.privacy_args.target_delta,
            num_steps=self.num_steps,
            sampling_probability=self.sampling_probability,
            eps_error=0.1,
        )

    def _init_extra_privacy_accounting(self) -> None:
        self.extra_rdp_accountant = None
        self.extra_prv_accountant = None
        self.combined_rdp_accountant = None
        self.combined_prv_accountant = None
        self.combined_prv_step_multiplier = 1

    def _build_privacy_budget_callback(self) -> PrivacyBudgetCallback:
        return PrivacyBudgetCallback(
            rdp_accountant=self.rdp_accountant,
            prv_accountant=self.prv_accountant,
            target_delta=self.privacy_args.target_delta,
            target_epsilon=self.privacy_args.target_epsilon,
            extra_rdp_accountant=self.extra_rdp_accountant,
            extra_prv_accountant=self.extra_prv_accountant,
            combined_rdp_accountant=self.combined_rdp_accountant,
            combined_prv_accountant=self.combined_prv_accountant,
            combined_prv_step_multiplier=self.combined_prv_step_multiplier,
        )

    def _get_additional_private_callbacks(self) -> List[TrainerCallback]:
        return []

    @property
    def sampling_probability(self) -> float:
        return (
            self.train_args.per_device_train_batch_size
            * self.train_args.world_size
            * self.train_args.gradient_accumulation_steps
            / len(self.author_mapping)
        )

    @property
    def num_steps(self) -> int:
        return int(
            self.train_args.num_train_epochs * (1 / self.sampling_probability + 1)
        )

    def create_optimizer(self):
        _ = super().create_optimizer()

        ghost_mode = bool(getattr(self.args, "ghost_mode", False))
        if self.args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
            if ghost_mode:
                optimizer_generator = (
                    opacus.optimizers.DistributedDPOptimizerFastGradientClipping
                )
            else:
                optimizer_generator = opacus.optimizers.DistributedDPOptimizer
        else:
            if ghost_mode:
                optimizer_generator = opacus.optimizers.DPOptimizerFastGradientClipping
            else:
                optimizer_generator = opacus.optimizers.DPOptimizer

        self.optimizer = optimizer_generator(
            optimizer=self.optimizer,
            noise_multiplier=self.privacy_args.noise_multiplier,
            max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
            expected_batch_size=self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps,
        )

        return self.optimizer

    def _infer_batch_size_from_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> int:
        for value in inputs.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        return 1

    def _run_batch_loss_metrics(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Optional[torch.Tensor]:
        get_batch_loss_metrics = getattr(self, "get_batch_loss_metrics", None)
        if not callable(get_batch_loss_metrics):
            return None

        try:
            sig = inspect.signature(get_batch_loss_metrics).parameters
            kwargs: Dict[str, Any] = {}
            if "train_eval" in sig:
                kwargs["train_eval"] = "train"
            maybe_loss_and_metrics = get_batch_loss_metrics(model, inputs, **kwargs)
            if (
                isinstance(maybe_loss_and_metrics, tuple)
                and len(maybe_loss_and_metrics) >= 1
            ):
                batch_loss = maybe_loss_and_metrics[0]
                metrics = (
                    maybe_loss_and_metrics[1]
                    if len(maybe_loss_and_metrics) > 1
                    else None
                )
                if isinstance(metrics, dict):
                    store_metrics = getattr(self, "store_metrics", None)
                    if callable(store_metrics):
                        try:
                            store_metrics(metrics, train_eval="train")
                        except TypeError:
                            store_metrics(metrics)
                if torch.is_tensor(batch_loss):
                    return batch_loss
            elif torch.is_tensor(maybe_loss_and_metrics):
                return maybe_loss_and_metrics
        except Exception:
            return None

        return None

    def _compute_loss_for_ghost(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_loss = self._run_batch_loss_metrics(model, inputs)
        if batch_loss is not None:
            return batch_loss

        compute_loss_params = inspect.signature(self.compute_loss).parameters
        if "num_items_in_batch" in compute_loss_params:
            return self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        return self.compute_loss(model, inputs)

    def _to_per_sample_loss(
        self,
        loss: torch.Tensor,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> torch.Tensor:
        if loss.ndim == 0:
            batch_size = self._infer_batch_size_from_inputs(inputs)
            if not getattr(self, "_ghost_scalar_loss_warned", False):
                print(
                    "[DP][Ghost] Warning: trainer returned scalar loss only; "
                    "falling back to scalar-based reweighting."
                )
                self._ghost_scalar_loss_warned = True
            return loss.repeat(batch_size)

        if loss.ndim > 1:
            return loss.reshape(loss.shape[0], -1).mean(dim=1)

        return loss

    def _ghost_training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            with self.compute_loss_context_manager():
                batch_loss = self._compute_loss_for_ghost(
                    model, inputs, num_items_in_batch=num_items_in_batch
                )
        except NotImplementedError as exc:
            # Opacus Ghost Clipping can fail when a step triggers repeated forwards
            # through the same trainable modules before backward (common in some RLHF losses).
            # In that case we gracefully fall back to Fast Gradient Clipping.
            if (
                "Parameter tying is not supported with Ghost Clipping" in str(exc)
                and getattr(model, "use_ghost_clipping", False)
            ):
                model.use_ghost_clipping = False
                if not getattr(self, "_ghost_fallback_warned", False):
                    print(
                        "[DP][Ghost] Warning: encountered unsupported repeated-forward "
                        "pattern; falling back to Fast Gradient Clipping for the "
                        "remaining steps."
                    )
                    self._ghost_fallback_warned = True
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                with self.compute_loss_context_manager():
                    batch_loss = self._compute_loss_for_ghost(
                        model, inputs, num_items_in_batch=num_items_in_batch
                    )
            else:
                raise

        if self.args.n_gpu > 1:
            batch_loss = batch_loss.mean()

        loss_per_sample = self._to_per_sample_loss(batch_loss, inputs)
        reduced_loss = loss_per_sample.mean()

        if self.use_apex:
            raise NotImplementedError("DP currently doesn't support this")

        reduced_loss.backward(retain_graph=True)
        self.optimizer.zero_grad()

        clipping_coef = model.get_clipping_coef().reshape(-1)
        per_sample_vec = loss_per_sample.reshape(-1)

        if per_sample_vec.numel() != clipping_coef.numel():
            if not getattr(self, "_ghost_shape_warned", False):
                print(
                    "[DP][Ghost] Warning: clipping coefficient and loss vector have "
                    "different sizes; applying aligned fallback."
                )
                self._ghost_shape_warned = True
            if per_sample_vec.numel() == 1:
                weighted_loss = per_sample_vec[0] * clipping_coef.mean() * clipping_coef.numel()
            else:
                n = min(per_sample_vec.numel(), clipping_coef.numel())
                weighted_loss = torch.sum(per_sample_vec[:n] * clipping_coef[:n])
        else:
            weighted_loss = torch.sum(per_sample_vec * clipping_coef)

        model.disable_hooks()
        weighted_loss.backward()
        model.enable_hooks()

        return reduced_loss.detach() / self.args.gradient_accumulation_steps

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            raise NotImplementedError("DP currently doesn't support this")

        if bool(getattr(self.args, "ghost_mode", False)):
            return self._ghost_training_step(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

        with self.compute_loss_context_manager():
            compute_loss_params = inspect.signature(self.compute_loss).parameters
            if "num_items_in_batch" in compute_loss_params:
                loss = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch
                )
            else:
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            raise NotImplementedError("DP currently doesn't support this")
        else:
            loss.backward()

        return loss.detach() / self.args.gradient_accumulation_steps

    def _get_train_sampler(self):
        return sampler.ShuffledAuthorSampler(
            author_mapping=self.author_mapping,
            batch_size=self.args.per_device_train_batch_size,
            world_size=self.args.world_size,
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )

        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class PrivateSFTTrainer(PrivateTrainer, SFTTrainer): ...


class PrivateDPOTrainer(PrivateTrainer, DPOTrainer): ...


class PrivateKTOTrainer(PrivateTrainer, DPOTrainer):
    """
    Tuple-private DP-KTO trainer.

    This keeps the protected unit aligned with a full (prompt, chosen, rejected)
    tuple while matching KTO's paper-style cyclic within-batch KL surrogate.

    For an ordered minibatch, we shift the chosen/rejected completions by one
    position with wrap-around, evaluate those donor completions under each
    prompt, clip both KL channels to [-C_z, C_z], average them inside the tuple,
    and release

        z_t = max(0, (1 / m_bar) * sum_i u_i + N(0, sigma_z^2 * (3 C_z / m_bar)^2)),

    where the factor 3 is the add/remove sensitivity of the cyclic sum and
    m_bar matches Opacus' fixed expected-batch-size denominator.
    """

    def __init__(
        self,
        *args,
        privacy_args: arguments.PrivacyArguments = None,
        author_mapping: Optional[Sequence[Sequence[int]]] = None,
        train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.kto_z_clip = float(getattr(privacy_args, "kto_z_clip", 1.0))
        # If z-noise is not set explicitly, we tie it to the epsilon-derived main
        # DP noise multiplier so the auxiliary scalar mechanism is not silently
        # left at a stale hard-coded value.
        self._requested_kto_z_noise_multiplier = getattr(
            privacy_args, "kto_z_noise_multiplier", None
        )
        self.kto_z_noise_multiplier = None
        self.kto_joint_noise_multiplier = None

        super().__init__(
            *args,
            privacy_args=privacy_args,
            author_mapping=author_mapping,
            train_dataset=train_dataset,
            **kwargs,
        )
        self.desirable_weight = float(getattr(self.args, "desirable_weight", 1.0))
        self.undesirable_weight = float(getattr(self.args, "undesirable_weight", 1.0))

    def _init_extra_privacy_accounting(self) -> None:
        super()._init_extra_privacy_accounting()
        # DP-KTO is the composition of two mechanisms on the same sampled batch:
        #   1. the standard Opacus gradient mechanism
        #   2. the cyclic scalar Gaussian release for z_t
        # We log the two components separately, but the privacy target is tracked
        # with a single accountant for the composed step so subsampling
        # amplification is applied once to the joint mechanism.
        if self._requested_kto_z_noise_multiplier is not None:
            self.kto_z_noise_multiplier = float(self._requested_kto_z_noise_multiplier)
        else:
            self.kto_z_noise_multiplier = float(self.privacy_args.noise_multiplier)
        self.kto_joint_noise_multiplier = self._joint_noise_multiplier(
            self.privacy_args.noise_multiplier
        )
        self.extra_rdp_accountant = RDPAccountant()
        self.extra_prv_accountant = PRVAccountant(
            noise_multiplier=self.kto_z_noise_multiplier,
            sampling_probability=self.sampling_probability,
            delta=self.privacy_args.target_delta,
            eps_error=0.1,
            max_compositions=self.num_steps,
        )
        self.combined_rdp_accountant = RDPAccountant()
        self.combined_prv_accountant = PRVAccountant(
            noise_multiplier=self.kto_joint_noise_multiplier,
            sampling_probability=self.sampling_probability,
            delta=self.privacy_args.target_delta,
            eps_error=0.1,
            max_compositions=self.num_steps,
        )
        self.combined_prv_step_multiplier = 1

    def _get_additional_private_callbacks(self) -> List[TrainerCallback]:
        return [
            AdditionalPrivacyBudgetCallback(
                accountant_steps=[
                    (
                        self.extra_rdp_accountant,
                        self.kto_z_noise_multiplier,
                        self.sampling_probability,
                        1,
                    ),
                    (
                        self.combined_rdp_accountant,
                        self.kto_joint_noise_multiplier,
                        self.sampling_probability,
                        1,
                    ),
                ],
            )
        ]

    def _find_noise_multiplier(self) -> float:
        target_epsilon = self.privacy_args.target_epsilon
        target_delta = self.privacy_args.target_delta
        if target_epsilon is None:
            raise ValueError("Combined DP-KTO calibration requires target_epsilon.")

        def compute_epsilon(main_noise_multiplier: float) -> float:
            accountant = RDPAccountant()
            for _ in range(self.num_steps):
                accountant.step(
                    noise_multiplier=self._joint_noise_multiplier(main_noise_multiplier),
                    sample_rate=self.sampling_probability,
                )
            return float(accountant.get_epsilon(target_delta))

        mu_max = 100.0
        mu_right = 1.0
        eps_right = float("inf")
        while eps_right > target_epsilon:
            mu_right *= math.sqrt(2.0)
            eps_right = compute_epsilon(mu_right)
            if mu_right > mu_max:
                raise RuntimeError(
                    "Finding a suitable combined DP-KTO noise multiplier did not converge."
                )

        mu_left = mu_right
        eps_left = eps_right
        while eps_left < target_epsilon:
            mu_left /= math.sqrt(2.0)
            eps_left = compute_epsilon(mu_left)

        for _ in range(40):
            mu_mid = 0.5 * (mu_left + mu_right)
            eps_mid = compute_epsilon(mu_mid)
            if eps_mid > target_epsilon:
                mu_left = mu_mid
            else:
                mu_right = mu_mid

        return mu_right

    def _joint_noise_multiplier(self, main_noise_multiplier: float) -> float:
        z_noise_multiplier = (
            self._requested_kto_z_noise_multiplier
            if self._requested_kto_z_noise_multiplier is not None
            else main_noise_multiplier
        )
        return 1.0 / math.sqrt(
            (1.0 / (main_noise_multiplier**2)) + (1.0 / (z_noise_multiplier**2))
        )

    def _expected_batch_size(self) -> float:
        expected_batch_size = float(
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        )
        if expected_batch_size <= 0:
            raise ValueError("DP-KTO requires a positive expected batch size.")
        return expected_batch_size

    def _z_sensitivity(self) -> float:
        # For the cyclic KTO estimator, deleting an interior tuple changes two
        # old KL terms and one new bridge term, so the clipped tuple-sum
        # sensitivity is 3 * C_z. We divide by Opacus' fixed denominator m_bar.
        return 3.0 * self.kto_z_clip / self._expected_batch_size()

    def _cyclic_reference_contrib(
        self,
        donor_chosen_logratios: torch.FloatTensor,
        donor_rejected_logratios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        clipped_chosen = torch.clamp(
            donor_chosen_logratios.detach(), -self.kto_z_clip, self.kto_z_clip
        )
        clipped_rejected = torch.clamp(
            donor_rejected_logratios.detach(), -self.kto_z_clip, self.kto_z_clip
        )
        return 0.5 * (clipped_chosen + clipped_rejected)

    def _build_cyclic_batch(
        self, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Dict[str, Union[List, torch.LongTensor]]:
        cyclic_batch = dict(batch)
        for key in (
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
        ):
            if key not in batch:
                raise KeyError(f"DP-KTO cyclic estimator requires `{key}` in the batch.")
            cyclic_batch[key] = torch.roll(batch[key], shifts=-1, dims=0)
        return cyclic_batch

    def _compute_cyclic_logratios(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        cyclic_batch = self._build_cyclic_batch(batch)
        with torch.no_grad():
            policy_output = self.concatenated_forward(
                model, cyclic_batch, is_ref_model=True
            )
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(cyclic_batch)

        donor_chosen_logratios = policy_output["chosen_logps"] - ref_chosen_logps
        donor_rejected_logratios = policy_output["rejected_logps"] - ref_rejected_logps
        return donor_chosen_logratios, donor_rejected_logratios

    def _release_private_reference_point(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        *,
        train_eval: str,
    ) -> torch.FloatTensor:
        # The z-release is detached on purpose: KTO treats the reference point as
        # a constant for the current step, and the DP fix privatizes that release
        # instead of backpropagating through the shared cyclic batch statistic.
        donor_chosen_logratios, donor_rejected_logratios = self._compute_cyclic_logratios(
            model, batch
        )
        tuple_contrib = self._cyclic_reference_contrib(
            donor_chosen_logratios, donor_rejected_logratios
        )
        z_mean = tuple_contrib.sum() / self._expected_batch_size()
        if train_eval != "train":
            return z_mean.clamp(min=0.0).detach()

        sensitivity = self._z_sensitivity()
        noise_std = self.kto_z_noise_multiplier * sensitivity
        noise = torch.randn((), device=z_mean.device, dtype=z_mean.dtype) * noise_std
        return (z_mean + noise).clamp(min=0.0).detach()

    def kto_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        *,
        z_priv: torch.FloatTensor,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        # The private z_t release is shared across the minibatch as in KTO, and
        # Opacus separately clips/noises the resulting tuple gradient in the
        # standard DP-SGD path.
        chosen_losses = torch.zeros_like(chosen_logratios)
        rejected_losses = torch.zeros_like(rejected_logratios)
        if chosen_logratios.numel() > 0:
            chosen_losses = self.desirable_weight * (
                1.0 - torch.sigmoid(self.beta * (chosen_logratios - z_priv))
            )
        if rejected_logratios.numel() > 0:
            rejected_losses = self.undesirable_weight * torch.sigmoid(
                self.beta * (rejected_logratios - z_priv)
            )

        losses = chosen_losses + rejected_losses
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: str = "train",
    ):
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        # We keep the paired DPO-style forward path because it naturally aligns
        # with tuple-private records (prompt, chosen, rejected).
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        z_priv = self._release_private_reference_point(
            model, batch, train_eval=train_eval
        )
        losses, chosen_rewards, rejected_rewards = self.kto_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            z_priv=z_priv,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}kto/z_priv"] = z_priv.detach().cpu()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (
            chosen_rewards - rejected_rewards
        ).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = (
            model_output["rejected_logps"].detach().mean().cpu()
        )
        metrics[f"{prefix}logits/chosen"] = (
            model_output["mean_chosen_logits"].detach().cpu()
        )
        metrics[f"{prefix}logits/rejected"] = (
            model_output["mean_rejected_logits"].detach().cpu()
        )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = model_output["aux_loss"].detach().cpu()

        return losses.mean(), metrics


class PrivateORPOTrainer(PrivateTrainer, ORPOTrainer): ...
