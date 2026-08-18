#!/usr/bin/env python3
"""Run the real Flax/Tunix recomputation and calibration loss on a TPU."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from calibration_aware_grpo.answers import extract_last_boxed_answer
from calibration_aware_grpo.jax_tunix import (
    compute_dual_objective_pg_loss,
    prepare_objective_branches,
)
from calibration_aware_grpo.tunix_runtime import TunixActorRecompute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a JSON result. Parent directories must exist.",
    )
    parser.add_argument(
        "--allow-non-tpu",
        action="store_true",
        help="Allow a local CPU/GPU preflight. Omit this for TPU evidence.",
    )
    return parser.parse_args()


def verifier(
    *,
    gold_answers: list[str],
    completion_texts: list[str],
) -> list[float]:
    return [
        float(extract_last_boxed_answer(completion) == gold)
        for gold, completion in zip(gold_answers, completion_texts, strict=True)
    ]


def build_model() -> Any:
    try:
        from flax import nnx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the `tunix` optional dependency first") from exc

    class TinyCausalLM(nnx.Module):
        def __init__(self, *, rngs: Any) -> None:
            self.embedding = nnx.Embed(32, 16, rngs=rngs)
            self.projection = nnx.Linear(16, 32, rngs=rngs)

        def __call__(
            self,
            tokens: Any,
            *,
            positions: Any,
            attention_mask: Any,
            cache: Any,
        ) -> Any:
            del positions, attention_mask, cache
            return self.projection(self.embedding(tokens))

    return TinyCausalLM(rngs=nnx.Rngs(0))


def block(value: Any) -> Any:
    return jax.tree.map(
        lambda leaf: (
            leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf
        ),
        value,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    backend = jax.default_backend()
    if backend != "tpu" and not args.allow_non_tpu:
        raise RuntimeError(f"expected a TPU backend, got {backend!r}")
    if args.output is not None and not args.output.parent.is_dir():
        raise ValueError(f"output parent does not exist: {args.output.parent}")

    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.asarray(devices), ("data",))
    model = build_model()
    prompt_tokens = jnp.asarray(
        [[1, 2, 3], [1, 2, 4], [1, 5, 3], [6, 2, 3]],
        dtype=jnp.int32,
    )
    prompt_mask = jnp.ones_like(prompt_tokens, dtype=jnp.bool_)
    completion_tokens = jnp.asarray(
        [[7, 8, 9, 10], [7, 8, 11, 10], [7, 12, 9, 10], [13, 8, 9, 10]],
        dtype=jnp.int32,
    )
    completion_mask = jnp.ones_like(completion_tokens, dtype=jnp.bool_)

    compile_started = time.time()
    with mesh:
        outputs = TunixActorRecompute(
            model=model,
            data_sharding_axis=("data",),
            micro_batch_size=2,
        ).outputs(
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            completion_tokens=completion_tokens,
            completion_mask=completion_mask,
        )
        block(outputs)
    recompute_seconds = time.time() - compile_started

    completion_texts = [r"\boxed{1}", r"\boxed{1}", r"\boxed{1}", r"\boxed{2}"]
    branches = prepare_objective_branches(
        completion_logits=outputs.completion_logits,
        completion_mask=completion_mask,
        completion_texts=completion_texts,
        gold_answers=["2"],
        group_size=4,
        verifier=verifier,
        mode="raw_signed_oc",
        calibration_lambda=0.25,
        overconfidence_lambda=0.1,
    )

    initial_log_ratios = jnp.linspace(
        -0.3,
        0.3,
        num=completion_mask.size,
        dtype=jnp.float32,
    ).reshape(completion_mask.shape)

    def loss_fn(log_ratios: Any) -> Any:
        ratios = jnp.exp(log_ratios)
        clipped = jnp.clip(ratios, 0.8, 1.2)
        return compute_dual_objective_pg_loss(
            coef_1=ratios,
            coef_2=clipped,
            completion_mask=completion_mask,
            branches=branches,
        ).loss

    train_step = jax.jit(jax.value_and_grad(loss_fn))
    train_started = time.time()
    loss, gradient = block(train_step(initial_log_ratios))
    updated = initial_log_ratios - 1e-2 * gradient
    block(updated)
    train_seconds = time.time() - train_started

    loss_value = float(loss)
    gradient_norm = float(jnp.linalg.norm(gradient))
    parameter_delta = float(jnp.max(jnp.abs(updated - initial_log_ratios)))
    if not math.isfinite(loss_value):
        raise RuntimeError(f"non-finite loss: {loss_value}")
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError(f"invalid gradient norm: {gradient_norm}")
    if parameter_delta <= 0:
        raise RuntimeError("optimizer update did not change the policy coefficients")
    if not bool(jnp.any(branches.advantages_calibration != 0)):
        raise RuntimeError("calibration branch was not exercised")
    if not bool(jnp.any(branches.advantages_overconfidence != 0)):
        raise RuntimeError("overconfidence branch was not exercised")

    receipt = {
        "schema": "calibration_aware_grpo.tpu_runtime_smoke.v1",
        "status": "passed",
        "backend": backend,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "device_count": len(devices),
        "device_kinds": sorted({device.device_kind for device in devices}),
        "checks": {
            "flax_nnx_model": True,
            "tunix_input_sharding": True,
            "teacher_forced_recompute": True,
            "microbatch_count": 2,
            "calibration_branch": True,
            "overconfidence_branch": True,
            "jitted_loss_and_gradient": True,
            "optimizer_update": True,
        },
        "metrics": {
            "loss": loss_value,
            "gradient_norm": gradient_norm,
            "parameter_max_delta": parameter_delta,
            "recompute_seconds": round(recompute_seconds, 3),
            "train_step_seconds": round(train_seconds, 3),
            "total_seconds": round(time.time() - started, 3),
        },
    }
    serialized = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
