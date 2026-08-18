from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibration_aware_grpo.tunix_runtime import (
    TunixActorRecompute,
    TunixLearnerPolicyRecompute,
    TunixRoles,
    TunixRuntimeOps,
    build_prompt_completion_inputs,
    compute_teacher_forced_outputs,
    resolve_policy_recompute_source,
)


class FakeModel:
    def __call__(self, input_tokens, *, positions, attention_mask, cache):
        del positions, attention_mask, cache
        vocabulary = 5
        offsets = jnp.arange(vocabulary, dtype=jnp.float32)
        logits = input_tokens[..., None].astype(jnp.float32) + offsets
        return logits, None


class FakeRollout:
    def __init__(self, model):
        self._model = model
        self.updated = []

    def model(self):
        return self._model

    def update_params(self, state):
        self.updated.append(state)


def chunk_slices(stop: int, step: int):
    return iter(slice(start, min(start + step, stop)) for start in range(0, stop, step))


def fake_ops() -> TunixRuntimeOps:
    return TunixRuntimeOps(
        split_model=lambda model: ("graph", model),
        merge_model=lambda graph, state: state if graph == "graph" else None,
        model_state=lambda model: {"model_id": id(model)},
        shard_input=lambda array, axis: array if axis == "data" else None,
        chunk_slices=chunk_slices,
        build_positions_from_mask=lambda mask: jnp.cumsum(mask, axis=-1) - 1,
        make_causal_attn_mask=lambda mask: mask[:, None, None, :],
        selective_log_softmax=lambda logits, targets: jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            targets[..., None],
            axis=-1,
        )[..., 0],
        stop_gradient=jax.lax.stop_gradient,
    )


def tensor_fixture():
    return {
        "prompt_tokens": jnp.asarray([[0, 2], [0, 3], [4, 1]], dtype=jnp.int32),
        "prompt_mask": jnp.asarray([[0, 1], [0, 1], [1, 1]], dtype=jnp.int32),
        "completion_tokens": jnp.asarray([[1, 2], [2, 3], [3, 4]], dtype=jnp.int32),
        "completion_mask": jnp.asarray([[1, 1], [1, 0], [1, 1]], dtype=jnp.int32),
    }


def test_prompt_completion_inputs_use_explicit_masks():
    fixture = tensor_fixture()
    tokens, positions, attention_mask = build_prompt_completion_inputs(
        **fixture,
        ops=fake_ops(),
    )
    np.testing.assert_array_equal(
        tokens,
        np.asarray([[0, 2, 1, 2], [0, 3, 2, 3], [4, 1, 3, 4]]),
    )
    np.testing.assert_array_equal(
        positions,
        np.asarray([[-1, 0, 1, 2], [-1, 0, 1, 1], [0, 1, 2, 3]]),
    )
    assert attention_mask.shape == (3, 1, 1, 4)


def test_teacher_forced_outputs_align_completion_targets():
    fixture = tensor_fixture()
    output = compute_teacher_forced_outputs(
        graphdef="graph",
        state=FakeModel(),
        ops=fake_ops(),
        **fixture,
    )

    assert output.completion_logits.shape == (3, 2, 5)
    assert output.per_token_logps.shape == (3, 2)
    assert output.sequence_self_certainty.shape == (3,)
    expected_logits = FakeModel()(
        jnp.concatenate(
            [fixture["prompt_tokens"], fixture["completion_tokens"]],
            axis=1,
        ),
        positions=None,
        attention_mask=None,
        cache=None,
    )[0][:, 1:3, :]
    np.testing.assert_allclose(output.completion_logits, expected_logits)
    assert output.sequence_self_certainty[1] > 0


def test_actor_recompute_microbatches_match_direct_forward():
    fixture = tensor_fixture()
    actor = TunixActorRecompute(
        model=FakeModel(),
        data_sharding_axis="data",
        micro_batch_size=2,
        ops=fake_ops(),
    )
    microbatched = actor.outputs(**fixture)
    direct = compute_teacher_forced_outputs(
        graphdef="graph",
        state=actor.model,
        ops=fake_ops(),
        **fixture,
    )
    np.testing.assert_allclose(microbatched.per_token_logps, direct.per_token_logps)
    np.testing.assert_allclose(microbatched.completion_logits, direct.completion_logits)
    np.testing.assert_allclose(
        microbatched.sequence_self_certainty,
        direct.sequence_self_certainty,
    )
    np.testing.assert_allclose(actor(**fixture), direct.completion_logits)


def fake_learner(*, engine: str, async_rollout: bool = False, offload: bool = False):
    model = FakeModel()
    rollout = FakeRollout(model)
    events = []

    @contextmanager
    def mesh(role):
        events.append(("mesh_enter", role))
        try:
            yield
        finally:
            events.append(("mesh_exit", role))

    cluster = SimpleNamespace(
        cluster_config=SimpleNamespace(
            rollout_engine=engine,
            offload_to_cpu=offload,
            training_config=SimpleNamespace(data_sharding_axis="data"),
        ),
        rollout=rollout,
        actor_trainer=SimpleNamespace(model=model),
        _get_mesh_and_logical_axis_rules_cm=mesh,
        _maybe_load_model_from_cpu=lambda current, role: events.append(("load", role)),
        _maybe_offload_model_to_cpu=lambda current, role: events.append(
            ("offload", role)
        ),
    )
    learner = SimpleNamespace(
        rl_cluster=cluster,
        can_enable_async_rollout=async_rollout,
        _compute_logps_micro_batch_size=1,
        algo_config=SimpleNamespace(num_generations=2),
    )
    return learner, events


def test_source_resolution_matches_rollout_engine_safety_rules():
    roles = TunixRoles(actor="actor", rollout="rollout")
    vanilla, _ = fake_learner(engine="vanilla")
    source = resolve_policy_recompute_source(vanilla, roles=roles)
    assert source.name == "rollout"
    assert source.role == "rollout"
    assert source.sync_rollout_params_when_offloaded is True

    vllm, _ = fake_learner(engine="vllm")
    source = resolve_policy_recompute_source(vllm, roles=roles)
    assert source.name == "actor"
    assert source.role == "actor"

    async_vllm, _ = fake_learner(engine="vllm", async_rollout=True)
    with pytest.raises(ValueError, match="asynchronous rollout"):
        resolve_policy_recompute_source(async_vllm, roles=roles)

    offloaded_vllm, _ = fake_learner(engine="vllm", offload=True)
    with pytest.raises(ValueError, match="offload_to_cpu"):
        resolve_policy_recompute_source(offloaded_vllm, roles=roles)


def test_learner_adapter_enters_mesh_and_resynchronizes_offloaded_rollout():
    learner, events = fake_learner(engine="vanilla", offload=True)
    adapter = TunixLearnerPolicyRecompute.from_learner(
        learner,
        roles=TunixRoles(actor="actor", rollout="rollout"),
        ops=fake_ops(),
    )
    output = adapter(**tensor_fixture())

    assert output.shape == (3, 2, 5)
    assert events == [
        ("mesh_enter", "rollout"),
        ("load", "rollout"),
        ("offload", "rollout"),
        ("mesh_exit", "rollout"),
    ]
    assert len(learner.rl_cluster.rollout.updated) == 2
