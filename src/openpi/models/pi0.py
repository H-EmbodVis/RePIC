import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def _picard_prefix_integrate(sample, velocities, deltas):
    increments = velocities * deltas.reshape((-1,) + (1,) * (velocities.ndim - 1))
    return jnp.concatenate([sample[None], sample[None] + jnp.cumsum(increments, axis=0)], axis=0)


def _picard_stride(x_before, x_after, time_nodes, window_size, tol, action_steps=None):
    diff = x_after - x_before
    if action_steps is not None:
        diff = diff[..., :action_steps, :]
    error = jnp.max(jnp.mean(jnp.square(diff), axis=(-2, -1)), axis=-1)
    threshold = jnp.square(tol * time_nodes)
    indices = jnp.arange(x_before.shape[0], dtype=jnp.int32)
    valid = (indices > 0) & (indices < window_size)
    ratio = jnp.where(valid, error / jnp.maximum(threshold, jnp.finfo(jnp.float32).tiny), 0.0)
    violations = valid & (error > threshold)
    stride = jnp.min(jnp.where(violations, indices, window_size))
    return stride, error, threshold, ratio


def _picard_noise_aligned_warm_trajectory(init_trajectory, time_nodes, ref, replan_steps):
    horizon = init_trajectory.shape[2]
    tail_steps = horizon - replan_steps
    sigma = jnp.concatenate([time_nodes, jnp.zeros((1,), dtype=time_nodes.dtype)])
    warm = init_trajectory + sigma[:, None, None, None] * (ref[None] - init_trajectory[0:1])
    source = init_trajectory[:, :, replan_steps:, :]
    shifted = source + sigma[:, None, None, None] * (ref[None, :, :tail_steps, :] - source[0:1])
    return warm.at[:, :, :tail_steps, :].set(shifted)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        repic: bool = False,
        picard_window: int = 10,
        picard_tol: float = 0.05,
        warm_start_trajectory: at.Float[at.Array, "s b ah ad"] | None = None,
        warm_start_replan_steps: int | None = None,
        threshold_action_steps: int | None = None,
        return_sampling_stats: bool = False,
        return_sampling_trace: bool = False,
    ) -> _model.Actions | dict:
        if return_sampling_trace and not return_sampling_stats:
            raise ValueError("return_sampling_trace requires return_sampling_stats")
        if repic and not self.pi05:
            raise ValueError("RePIC is only supported by pi0.5 models")
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if repic:
            return self._sample_actions_picard(
                observation,
                noise,
                prefix_mask,
                kv_cache,
                num_steps=int(num_steps),
                window=int(picard_window),
                tol=float(picard_tol),
                warm_start_trajectory=warm_start_trajectory,
                warm_start_replan_steps=warm_start_replan_steps,
                threshold_action_steps=threshold_action_steps,
                return_sampling_stats=return_sampling_stats,
                return_sampling_trace=return_sampling_trace,
            )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _predict_velocity_with_cache(self, observation, x_t, time, prefix_mask, kv_cache):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=positions,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond]
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def _sample_actions_picard(
        self,
        observation,
        noise,
        prefix_mask,
        kv_cache,
        *,
        num_steps,
        window,
        tol,
        warm_start_trajectory,
        warm_start_replan_steps,
        threshold_action_steps,
        return_sampling_stats,
        return_sampling_trace,
    ):
        if num_steps <= 0 or window <= 0:
            raise ValueError("num_steps and picard_window must be positive")
        window = min(int(window), int(num_steps))
        batch_size = noise.shape[0]
        threshold_steps = self.action_horizon if threshold_action_steps is None else int(threshold_action_steps)
        if not 1 <= threshold_steps <= self.action_horizon:
            raise ValueError("threshold_action_steps must be in [1, action_horizon]")
        if warm_start_trajectory is not None:
            if warm_start_replan_steps is None:
                raise ValueError("warm_start_replan_steps is required with warm_start_trajectory")
            expected = (num_steps + 1, *noise.shape)
            if tuple(warm_start_trajectory.shape) != expected:
                raise ValueError(f"warm_start_trajectory must have shape {expected}, got {warm_start_trajectory.shape}")
            if not 0 < int(warm_start_replan_steps) < self.action_horizon:
                raise ValueError("warm_start_replan_steps must be in [1, action_horizon)")

        dt = -1.0 / float(num_steps)
        time_nodes = 1.0 + jnp.arange(num_steps + window + 1, dtype=noise.dtype) * dt
        time_nodes = jnp.where(jnp.arange(time_nodes.shape[0]) <= num_steps, time_nodes, 0.0)
        deltas = jnp.concatenate(
            [jnp.full((num_steps,), dt, dtype=noise.dtype), jnp.zeros((window,), dtype=noise.dtype)]
        )
        if warm_start_trajectory is None:
            draft_velocity = self._predict_velocity_with_cache(
                observation, noise, jnp.ones((batch_size,), dtype=noise.dtype), prefix_mask, kv_cache
            ).astype(noise.dtype)
            initial_nfe = 1
            initial_filled_upto = 0
        else:
            draft_velocity = jnp.zeros_like(noise)
            initial_nfe = 0
            initial_filled_upto = num_steps

        repeated_observation = jax.tree.map(
            lambda x: jnp.tile(x, (window,) + (1,) * (x.ndim - 1)), observation
        )
        repeated_prefix_mask = jnp.tile(prefix_mask, (window, 1))
        repeated_kv_cache = jax.tree.map(
            lambda x: jnp.tile(x, (1, window) + (1,) * (x.ndim - 2)), kv_cache
        )
        trajectory = jnp.zeros((num_steps + window + 1, *noise.shape), dtype=noise.dtype)
        trajectory = trajectory.at[0].set(noise)
        if warm_start_trajectory is not None:
            warm = _picard_noise_aligned_warm_trajectory(
                warm_start_trajectory.astype(noise.dtype), time_nodes[:num_steps], noise,
                int(warm_start_replan_steps)
            )
            trajectory = trajectory.at[: num_steps + 1].set(warm)

        round_valid = jnp.zeros((num_steps,), dtype=jnp.bool_)
        round_start = jnp.full((num_steps,), -1, dtype=jnp.int32)
        round_size = jnp.zeros((num_steps,), dtype=jnp.int32)
        round_stride = jnp.zeros((num_steps,), dtype=jnp.int32)
        round_max_ratio = jnp.zeros((num_steps,), dtype=jnp.float32)
        final_residual = jnp.zeros((batch_size, self.action_horizon), dtype=jnp.float32)
        if return_sampling_trace:
            trace = {
                "round_x_before": jnp.zeros((num_steps, window + 1, *noise.shape), dtype=jnp.float32),
                "round_x_after": jnp.zeros((num_steps, window + 1, *noise.shape), dtype=jnp.float32),
                "round_velocities": jnp.zeros((num_steps, window, *noise.shape), dtype=jnp.float32),
                "round_v_draft": jnp.zeros((num_steps, *noise.shape), dtype=jnp.float32),
                "round_v_draft_valid": jnp.zeros((num_steps,), dtype=jnp.bool_),
                "round_used_v_free": jnp.zeros((num_steps,), dtype=jnp.bool_),
            }
        else:
            trace = {}

        def cond(carry):
            return carry[0] < num_steps

        def body(carry):
            (
                step_index, filled_upto, num_rounds, nfe, trajectory, v_free,
                round_valid, round_start, round_size, round_stride, round_max_ratio,
                final_residual, num_truncated, max_error_ratio, trace,
            ) = carry
            window_size = jnp.minimum(window, num_steps - step_index)
            right_edge = step_index + window_size
            need_fill = filled_upto < right_edge
            used_v_free = num_rounds > 0
            v_draft = jnp.where(used_v_free, v_free, draft_velocity)
            fill_base = trajectory[filled_upto]
            fill_time = time_nodes[filled_upto]

            def fill_one(j, xs):
                index = filled_upto + j + 1
                value = fill_base + (time_nodes[index] - fill_time) * v_draft
                return jax.lax.cond(
                    need_fill & (index <= right_edge),
                    lambda current: current.at[index].set(value),
                    lambda current: current,
                    xs,
                )

            trajectory = jax.lax.fori_loop(0, window, fill_one, trajectory)
            filled_upto = jnp.maximum(filled_upto, right_edge)
            window_states = jax.lax.dynamic_slice(
                trajectory, (step_index, 0, 0, 0),
                (window, batch_size, self.action_horizon, self.action_dim)
            )
            window_times = jax.lax.dynamic_slice(time_nodes, (step_index,), (window,))
            flat_states = window_states.reshape((window * batch_size, self.action_horizon, self.action_dim))
            flat_times = jnp.repeat(window_times, batch_size)
            flat_velocities = self._predict_velocity_with_cache(
                repeated_observation, flat_states, flat_times, repeated_prefix_mask, repeated_kv_cache
            ).astype(noise.dtype)
            velocities = flat_velocities.reshape((window, batch_size, self.action_horizon, self.action_dim))
            window_deltas = jax.lax.dynamic_slice(deltas, (step_index,), (window,))
            x_before = jax.lax.dynamic_slice(
                trajectory, (step_index, 0, 0, 0),
                (window + 1, batch_size, self.action_horizon, self.action_dim)
            )
            x_after = _picard_prefix_integrate(trajectory[step_index], velocities, window_deltas)
            stride, _, _, ratios = _picard_stride(
                x_before, x_after,
                jax.lax.dynamic_slice(time_nodes, (step_index,), (window + 1,)),
                window_size, tol, threshold_steps
            )
            indices = jnp.arange(window + 1, dtype=jnp.int32)
            valid_states = (indices > 0) & (indices <= window_size)

            def write_one(j, xs):
                return jax.lax.cond(
                    valid_states[j],
                    lambda current: current.at[step_index + j].set(x_after[j]),
                    lambda current: current,
                    xs,
                )

            trajectory = jax.lax.fori_loop(1, window + 1, write_one, trajectory)
            v_free = velocities[stride - 1]
            valid_ratios = (indices > 0) & (indices < window_size)
            window_max_ratio = jnp.max(jnp.where(valid_ratios, ratios, 0.0))
            truncated = stride < window_size
            residual_index = window_size
            clean_residual = jnp.sqrt(
                jnp.mean(jnp.square(x_after[residual_index] - x_before[residual_index]), axis=-1)
            )
            final_residual = jax.lax.cond(
                step_index + window_size == num_steps,
                lambda _: clean_residual.astype(jnp.float32),
                lambda _: final_residual,
                operand=None,
            )
            round_valid = round_valid.at[num_rounds].set(True)
            round_start = round_start.at[num_rounds].set(step_index)
            round_size = round_size.at[num_rounds].set(window_size)
            round_stride = round_stride.at[num_rounds].set(stride)
            round_max_ratio = round_max_ratio.at[num_rounds].set(window_max_ratio)
            if return_sampling_trace:
                trace = {
                    **trace,
                    "round_x_before": trace["round_x_before"].at[num_rounds].set(x_before.astype(jnp.float32)),
                    "round_x_after": trace["round_x_after"].at[num_rounds].set(x_after.astype(jnp.float32)),
                    "round_velocities": trace["round_velocities"].at[num_rounds].set(velocities.astype(jnp.float32)),
                    "round_v_draft": trace["round_v_draft"].at[num_rounds].set(v_draft.astype(jnp.float32)),
                    "round_v_draft_valid": trace["round_v_draft_valid"].at[num_rounds].set(need_fill),
                    "round_used_v_free": trace["round_used_v_free"].at[num_rounds].set(used_v_free),
                }
            return (
                step_index + stride, filled_upto, num_rounds + 1, nfe + 1,
                trajectory, v_free, round_valid, round_start, round_size,
                round_stride, round_max_ratio, final_residual,
                num_truncated + truncated.astype(jnp.int32),
                jnp.maximum(max_error_ratio, window_max_ratio), trace,
            )

        carry = (
            jnp.asarray(0, dtype=jnp.int32), jnp.asarray(initial_filled_upto, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32), jnp.asarray(initial_nfe, dtype=jnp.int32),
            trajectory, jnp.zeros_like(noise), round_valid, round_start, round_size,
            round_stride, round_max_ratio, final_residual, jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32), trace,
        )
        (_, _, num_rounds, nfe, trajectory, _, round_valid, round_start, round_size,
         round_stride, round_max_ratio, final_residual, num_truncated, max_error_ratio, trace) = (
            jax.lax.while_loop(cond, body, carry)
        )
        actions = trajectory[num_steps]
        if not return_sampling_stats:
            return actions
        stats = {
            "nfe": nfe,
            "num_rounds": num_rounds,
            "num_steps": jnp.asarray(num_steps, dtype=jnp.int32),
            "picard_window": jnp.asarray(window, dtype=jnp.int32),
            "picard_tol": jnp.asarray(tol, dtype=jnp.float32),
            "mean_stride": jnp.asarray(num_steps, dtype=jnp.float32) / jnp.maximum(num_rounds, 1),
            "num_truncated_rounds": num_truncated,
            "truncation_rate": num_truncated.astype(jnp.float32) / jnp.maximum(num_rounds, 1),
            "max_error_ratio": max_error_ratio,
            "warm_start_used": jnp.asarray(warm_start_trajectory is not None),
            "warm_start_replan_steps": jnp.asarray(0 if warm_start_replan_steps is None else warm_start_replan_steps, dtype=jnp.int32),
            "threshold_action_steps": jnp.asarray(threshold_steps, dtype=jnp.int32),
            "final_clean_action_residual": final_residual,
            "final_trajectory": trajectory[: num_steps + 1],
            "round_valid": round_valid,
            "round_start": round_start,
            "round_size": round_size,
            "round_stride": round_stride,
            "round_max_error_ratio": round_max_ratio,
        }
        result = {"actions": actions, "sampling_stats": stats}
        if return_sampling_trace:
            result["sampling_trace"] = {
                "initial_noise": noise,
                "time_nodes": time_nodes[: num_steps + 1],
                "deltas": deltas[:num_steps],
                "final_trajectory": trajectory[: num_steps + 1].astype(jnp.float32),
                "normalized_actions": actions,
                **trace,
            }
        return result
