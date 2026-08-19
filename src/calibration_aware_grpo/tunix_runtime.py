"""Flax NNX/Tunix adapter for teacher-forced policy recomputation.

Imports remain lazy so the core package can be installed without the full Tunix
stack. Tests use small fake operations and models to exercise the control flow
on CPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from .jax_tunix import compute_self_certainty_from_logits, mean_token_scores

Array = Any


class ModelGetter(Protocol):
    def __call__(self) -> Any: ...


@dataclass(frozen=True)
class TunixRuntimeOps:
    """The narrow Flax/Tunix operation surface used by policy recomputation."""

    split_model: Callable[[Any], tuple[Any, Any]]
    merge_model: Callable[[Any, Any], Any]
    model_state: Callable[[Any], Any]
    shard_input: Callable[[Array, Any], Array]
    chunk_slices: Callable[[int, int], Iterator[slice]]
    build_positions_from_mask: Callable[[Array], Array]
    make_causal_attn_mask: Callable[[Array], Array]
    selective_log_softmax: Callable[[Array, Array], Array]
    stop_gradient: Callable[[Array], Array]


@dataclass(frozen=True)
class TunixRoles:
    """Tunix role constants, separated to keep tests framework-independent."""

    actor: Any
    rollout: Any


@dataclass(frozen=True)
class TeacherForcedOutputs:
    """Completion-token outputs from one teacher-forced model microbatch."""

    per_token_logps: Array
    completion_logits: Array
    sequence_self_certainty: Array


@dataclass(frozen=True)
class PolicyRecomputeSource:
    """Resolved model/mesh source whose weights must match rollout generation."""

    name: str
    role: Any
    model_getter: ModelGetter
    sync_rollout_params_when_offloaded: bool = False


def load_tunix_runtime_ops() -> TunixRuntimeOps:
    """Load the Flax/Tunix APIs used by the optional Tunix integration.

    Raises a focused import error instead of making Tunix a mandatory dependency
    of the core package.
    """

    try:
        from flax import nnx  # type: ignore[import-not-found]
        from tunix.rl import common as rl_common  # type: ignore[import-not-found]
        from tunix.rl import utils as rl_utils  # type: ignore[import-not-found]
        from tunix.sft import sharding_utils  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise ImportError(
            "Concrete Tunix recomputation requires the 'tunix' optional extra"
        ) from exc

    return TunixRuntimeOps(
        split_model=nnx.split,
        merge_model=nnx.merge,
        model_state=nnx.state,
        shard_input=sharding_utils.shard_input,
        chunk_slices=lambda stop, step: rl_utils.chunk_slices_by_size(
            stop=stop,
            step=step,
        ),
        build_positions_from_mask=rl_common.build_positions_from_mask,
        make_causal_attn_mask=rl_common.make_causal_attn_mask,
        selective_log_softmax=rl_common.selective_log_softmax,
        stop_gradient=jax.lax.stop_gradient,
    )


def load_tunix_roles() -> TunixRoles:
    try:
        from tunix.rl.rl_cluster import Role  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise ImportError(
            "Concrete Tunix recomputation requires the 'tunix' optional extra"
        ) from exc
    return TunixRoles(actor=Role.ACTOR, rollout=Role.ROLLOUT)


def build_prompt_completion_inputs(
    *,
    prompt_tokens: Array,
    prompt_mask: Array,
    completion_tokens: Array,
    completion_mask: Array,
    ops: TunixRuntimeOps,
) -> tuple[Array, Array, Array]:
    """Build model inputs without reconstructing masks from token IDs."""

    prompt_tokens = jnp.asarray(prompt_tokens)
    completion_tokens = jnp.asarray(completion_tokens)
    prompt_mask = jnp.asarray(prompt_mask, dtype=jnp.bool_)
    completion_mask = jnp.asarray(completion_mask, dtype=jnp.bool_)
    if prompt_tokens.ndim != 2 or completion_tokens.ndim != 2:
        raise ValueError("prompt_tokens and completion_tokens must be rank-2")
    if prompt_tokens.shape != prompt_mask.shape:
        raise ValueError("prompt_tokens and prompt_mask must have identical shapes")
    if completion_tokens.shape != completion_mask.shape:
        raise ValueError(
            "completion_tokens and completion_mask must have identical shapes"
        )
    if prompt_tokens.shape[0] != completion_tokens.shape[0]:
        raise ValueError("prompt and completion batches must have the same size")

    input_tokens = jnp.concatenate([prompt_tokens, completion_tokens], axis=1)
    input_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)
    positions = ops.build_positions_from_mask(input_mask)
    attention_mask = ops.make_causal_attn_mask(input_mask)
    return input_tokens, positions, attention_mask


def compute_teacher_forced_outputs(
    *,
    graphdef: Any,
    state: Any,
    prompt_tokens: Array,
    prompt_mask: Array,
    completion_tokens: Array,
    completion_mask: Array,
    ops: TunixRuntimeOps,
    stop_gradient: bool = True,
) -> TeacherForcedOutputs:
    """Teacher-force completions and retain exactly their predictive logits."""

    model = ops.merge_model(graphdef, state)
    input_tokens, positions, attention_mask = build_prompt_completion_inputs(
        prompt_tokens=prompt_tokens,
        prompt_mask=prompt_mask,
        completion_tokens=completion_tokens,
        completion_mask=completion_mask,
        ops=ops,
    )
    model_output = model(
        input_tokens,
        positions=positions,
        attention_mask=attention_mask,
        cache=None,
    )
    logits = model_output[0] if isinstance(model_output, tuple) else model_output
    logits = jnp.asarray(logits, dtype=jnp.float32)
    completion_length = int(jnp.asarray(completion_tokens).shape[1])
    if completion_length < 1:
        raise ValueError("completion_tokens must contain at least one token")
    if logits.ndim != 3 or logits.shape[:2] != input_tokens.shape:
        raise ValueError(
            "model logits must have shape [samples, prompt+completion, vocabulary]; "
            f"got {logits.shape} for inputs {input_tokens.shape}"
        )

    completion_logits = logits[:, -completion_length - 1 : -1, :]
    target_tokens = input_tokens[:, -completion_length:]
    if completion_logits.shape[1] != completion_length:
        raise ValueError(
            "model sequence is too short to align every completion token with a "
            "predictive logit"
        )
    per_token_logps = ops.selective_log_softmax(
        completion_logits,
        target_tokens,
    )
    if stop_gradient:
        per_token_logps = ops.stop_gradient(per_token_logps)
        completion_logits = ops.stop_gradient(completion_logits)

    token_self_certainty = compute_self_certainty_from_logits(completion_logits)
    sequence_self_certainty = mean_token_scores(
        token_self_certainty,
        completion_mask,
    )
    if stop_gradient:
        sequence_self_certainty = ops.stop_gradient(sequence_self_certainty)
    return TeacherForcedOutputs(
        per_token_logps=jnp.asarray(per_token_logps, dtype=jnp.float32),
        completion_logits=jnp.asarray(completion_logits, dtype=jnp.float32),
        sequence_self_certainty=jnp.asarray(
            sequence_self_certainty,
            dtype=jnp.float32,
        ),
    )


@dataclass
class TunixActorRecompute:
    """Microbatched actor forward pass implementing ``PolicyRecompute``."""

    model: Any
    data_sharding_axis: Any
    micro_batch_size: int | None = None
    ops: TunixRuntimeOps | None = None

    def _ops(self) -> TunixRuntimeOps:
        return self.ops or load_tunix_runtime_ops()

    def outputs(
        self,
        *,
        prompt_tokens: Array,
        prompt_mask: Array,
        completion_tokens: Array,
        completion_mask: Array,
    ) -> TeacherForcedOutputs:
        ops = self._ops()
        arrays = [
            jnp.asarray(prompt_tokens),
            jnp.asarray(prompt_mask),
            jnp.asarray(completion_tokens),
            jnp.asarray(completion_mask),
        ]
        batch_size = int(arrays[0].shape[0])
        if batch_size < 1:
            raise ValueError("cannot recompute an empty rollout batch")
        if any(int(array.shape[0]) != batch_size for array in arrays):
            raise ValueError("all recomputation tensors must share a batch size")
        step = self.micro_batch_size or batch_size
        if step < 1:
            raise ValueError("micro_batch_size must be positive")
        sharded = [ops.shard_input(array, self.data_sharding_axis) for array in arrays]
        graphdef, state = ops.split_model(self.model)
        outputs = [
            compute_teacher_forced_outputs(
                graphdef=graphdef,
                state=state,
                prompt_tokens=sharded[0][batch_slice],
                prompt_mask=sharded[1][batch_slice],
                completion_tokens=sharded[2][batch_slice],
                completion_mask=sharded[3][batch_slice],
                ops=ops,
            )
            for batch_slice in ops.chunk_slices(batch_size, step)
        ]
        if not outputs:
            raise RuntimeError("Tunix chunking returned no microbatches")
        return TeacherForcedOutputs(
            per_token_logps=jnp.concatenate(
                [output.per_token_logps for output in outputs], axis=0
            ),
            completion_logits=jnp.concatenate(
                [output.completion_logits for output in outputs], axis=0
            ),
            sequence_self_certainty=jnp.concatenate(
                [output.sequence_self_certainty for output in outputs], axis=0
            ),
        )

    def __call__(
        self,
        *,
        prompt_tokens: Array,
        prompt_mask: Array,
        completion_tokens: Array,
        completion_mask: Array,
    ) -> Array:
        """Return completion logits for ``prepare_objective_from_policy``."""

        return self.outputs(
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            completion_tokens=completion_tokens,
            completion_mask=completion_mask,
        ).completion_logits


def resolve_policy_recompute_source(
    learner: Any,
    *,
    roles: TunixRoles | None = None,
) -> PolicyRecomputeSource:
    """Select the rollout-matching model and reject unsafe vLLM layouts."""

    resolved_roles = roles or load_tunix_roles()
    cluster = learner.rl_cluster
    rollout_engine = cluster.cluster_config.rollout_engine
    if rollout_engine == "vanilla":
        return PolicyRecomputeSource(
            name="rollout",
            role=resolved_roles.rollout,
            model_getter=cluster.rollout.model,
            sync_rollout_params_when_offloaded=True,
        )
    if rollout_engine == "vllm":
        if bool(getattr(learner, "can_enable_async_rollout", False)):
            raise ValueError(
                "vLLM actor recomputation is unsafe when asynchronous rollout "
                "can use a different mesh or weight version"
            )
        if bool(getattr(cluster.cluster_config, "offload_to_cpu", False)):
            raise ValueError(
                "vLLM actor recomputation with offload_to_cpu is unsupported"
            )
        return PolicyRecomputeSource(
            name="actor",
            role=resolved_roles.actor,
            model_getter=lambda: cluster.actor_trainer.model,
        )
    raise ValueError(
        f"unsupported Tunix rollout engine for policy recomputation: {rollout_engine!r}"
    )


@dataclass
class TunixLearnerPolicyRecompute:
    """Mesh/offload-aware adapter constructed directly from a Tunix learner."""

    learner: Any
    source: PolicyRecomputeSource
    data_sharding_axis: Any
    micro_batch_size: int | None
    ops: TunixRuntimeOps | None = None

    @classmethod
    def from_learner(
        cls,
        learner: Any,
        *,
        roles: TunixRoles | None = None,
        ops: TunixRuntimeOps | None = None,
    ) -> TunixLearnerPolicyRecompute:
        cluster = learner.rl_cluster
        base_micro_batch_size = getattr(
            learner,
            "_compute_logps_micro_batch_size",
            None,
        )
        if base_micro_batch_size is not None:
            base_micro_batch_size = int(base_micro_batch_size) * int(
                learner.algo_config.num_generations
            )
        return cls(
            learner=learner,
            source=resolve_policy_recompute_source(learner, roles=roles),
            data_sharding_axis=(
                cluster.cluster_config.training_config.data_sharding_axis
            ),
            micro_batch_size=base_micro_batch_size,
            ops=ops,
        )

    def __call__(
        self,
        *,
        prompt_tokens: Array,
        prompt_mask: Array,
        completion_tokens: Array,
        completion_mask: Array,
    ) -> Array:
        cluster = self.learner.rl_cluster
        runtime_ops = self.ops or load_tunix_runtime_ops()
        offloaded = bool(getattr(cluster.cluster_config, "offload_to_cpu", False))
        with cluster._get_mesh_and_logical_axis_rules_cm(self.source.role):
            model = self.source.model_getter()
            cluster._maybe_load_model_from_cpu(model, self.source.role)
            if self.source.sync_rollout_params_when_offloaded and offloaded:
                cluster.rollout.update_params(runtime_ops.model_state(model))
            try:
                return TunixActorRecompute(
                    model=model,
                    data_sharding_axis=self.data_sharding_axis,
                    micro_batch_size=self.micro_batch_size,
                    ops=runtime_ops,
                )(
                    prompt_tokens=prompt_tokens,
                    prompt_mask=prompt_mask,
                    completion_tokens=completion_tokens,
                    completion_mask=completion_mask,
                )
            finally:
                cluster._maybe_offload_model_to_cpu(model, self.source.role)
                if self.source.sync_rollout_params_when_offloaded and offloaded:
                    cluster.rollout.update_params(runtime_ops.model_state(model))
