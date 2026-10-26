# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish

from .cross_attention_dit import DiT, SelfAttentionTransformer


def _noise_aligned_warm_start(
    cached: torch.Tensor, noise: torch.Tensor, replan_steps: int
) -> torch.Tensor:
    """Shift the cached denoising trajectory and align it to fresh noise."""
    num_steps = cached.shape[0] - 1
    sigma = 1.0 - torch.arange(num_steps + 1, device=cached.device, dtype=cached.dtype) / num_steps
    warm = cached + sigma[:, None, None, None] * (noise[None] - cached[:1])
    tail_steps = cached.shape[2] - replan_steps
    if tail_steps:
        source = cached[:, :, replan_steps : replan_steps + tail_steps]
        warm[:, :, :tail_steps] = source + sigma[:, None, None, None] * (
            noise[:, :tail_steps][None] - source[:1]
        )
    return warm


def _picard_stride(
    previous: torch.Tensor,
    updated: torch.Tensor,
    start_step: int,
    window_size: int,
    num_steps: int,
    tolerance: float,
    action_steps: int | None,
) -> tuple[int, list[float]]:
    """Choose the first residual-violating stride in a Picard window."""
    prefix = updated if action_steps is None else updated[..., :action_steps, :]
    previous_prefix = previous if action_steps is None else previous[..., :action_steps, :]
    residual = (prefix - previous_prefix).float()
    errors = residual.square().mean(dim=(-3, -2, -1))
    if window_size <= 1:
        return 1, []
    indices = torch.arange(1, window_size, device=errors.device, dtype=torch.long)
    sigma = (start_step + indices).to(errors.dtype) / float(num_steps)
    sigma = sigma.clamp_min(1.0 / float(num_steps))
    threshold = (tolerance * sigma).square()
    window_errors = errors[1:window_size]
    ratios = window_errors / threshold.clamp_min(torch.finfo(errors.dtype).tiny)
    stride = int(
        torch.where(window_errors > threshold, indices, torch.full_like(indices, window_size))
        .min()
        .item()
    )
    return stride, ratios.detach().cpu().tolist()


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.last_repic_trajectory = None
        self.last_repic_stats = None
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @torch.no_grad()
    def get_action_repic(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        *,
        picard_window: int = 4,
        picard_tol: float = 0.1,
        init_trajectory: torch.Tensor | None = None,
        warm_start_replan_steps: int = 1,
        threshold_action_steps: int | None = 1,
    ) -> BatchFeature:
        """Sample with RePIC's batched Picard refinement and trajectory reuse."""
        self.last_repic_trajectory = None
        self.last_repic_stats = None
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        dtype = vl_embs.dtype
        state_features = self.state_encoder(action_input.state, action_input.embodiment_id)

        num_steps = int(self.num_inference_timesteps)
        if num_steps <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {num_steps}")
        window = int(picard_window)
        if not 1 <= window <= num_steps:
            raise ValueError(f"picard_window must be in [1, {num_steps}], got {window}")
        tol = float(picard_tol)
        if tol < 0:
            raise ValueError(f"picard_tol must be non-negative, got {tol}")
        replan_steps = int(warm_start_replan_steps)
        if not 1 <= replan_steps <= self.action_horizon:
            raise ValueError(
                "warm_start_replan_steps must be in "
                f"[1, {self.action_horizon}], got {replan_steps}"
            )
        if threshold_action_steps is not None:
            threshold_action_steps = int(threshold_action_steps)
            if not 1 <= threshold_action_steps <= self.action_horizon:
                raise ValueError(
                    "threshold_action_steps must be in "
                    f"[1, {self.action_horizon}], got {threshold_action_steps}"
                )

        noise = torch.randn(
            (batch_size, self.action_horizon, self.action_dim), device=device, dtype=dtype
        )
        dt = 1.0 / float(num_steps)

        if init_trajectory is None:
            trajectory = [noise]
            filled_upto = 0
            init_mode = "linear"
        else:
            cached = init_trajectory.detach()
            if cached.ndim == 3:
                cached = cached.unsqueeze(1)
            expected_shape = (
                num_steps + 1,
                batch_size,
                self.action_horizon,
                self.action_dim,
            )
            if tuple(cached.shape) != expected_shape:
                raise ValueError(
                    f"init_trajectory must have shape {expected_shape}, got {tuple(cached.shape)}"
                )
            cached = cached.to(device=device, dtype=dtype)
            warm = _noise_aligned_warm_start(cached, noise, replan_steps)
            trajectory = [warm[index].contiguous() for index in range(num_steps + 1)]
            filled_upto = num_steps
            init_mode = "noise_aligned_history"

        def velocity(action_batch: torch.Tensor, timestep_batch: torch.Tensor) -> torch.Tensor:
            batch = int(action_batch.shape[0])
            if batch % batch_size:
                raise ValueError("Picard batch size must be a multiple of the input batch size")
            repeats = batch // batch_size
            embodiment = action_input.embodiment_id.repeat(repeats)
            action_features = self.action_encoder(action_batch, timestep_batch, embodiment)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], device=device, dtype=torch.long)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
            state_batch = state_features.repeat(repeats, 1, 1)
            sa_embs = torch.cat((state_batch, future_tokens, action_features), dim=1)
            mask = backbone_output.backbone_attention_mask
            if mask is not None:
                mask = mask.repeat(repeats, 1)
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs.repeat(repeats, 1, 1),
                encoder_attention_mask=mask,
                timestep=timestep_batch,
            )
            pred = self.action_decoder(model_output, embodiment)
            return pred[:, -self.action_horizon :]

        t = 0
        nfe = 0
        rounds = []
        stride_sum = 0
        v_free = None
        while t < num_steps:
            window_size = min(window, num_steps - t)
            while len(trajectory) < t + window_size + 1:
                trajectory.append(trajectory[-1].clone())

            if filled_upto < t + window_size:
                if v_free is None:
                    draft_t = torch.full(
                        (batch_size,),
                        int((t / num_steps) * self.num_timestep_buckets),
                        device=device,
                        dtype=torch.long,
                    )
                    draft_velocity = velocity(trajectory[t], draft_t)
                    nfe += 1
                else:
                    draft_velocity = v_free
                current = trajectory[filled_upto]
                for index in range(filled_upto, t + window_size):
                    current = current + dt * draft_velocity
                    trajectory[index + 1] = current
                filled_upto = t + window_size

            states = torch.cat(trajectory[t : t + window_size], dim=0)
            timestep_values = [
                int((index / num_steps) * self.num_timestep_buckets)
                for index in range(t, t + window_size)
            ]
            timesteps = torch.tensor(timestep_values, device=device, dtype=torch.long).repeat_interleave(
                batch_size
            )
            velocities = velocity(states, timesteps).reshape(
                window_size, batch_size, self.action_horizon, self.action_dim
            )
            nfe += 1

            updated = [trajectory[t]]
            current = trajectory[t]
            for index in range(window_size):
                current = current + dt * velocities[index]
                updated.append(current)

            previous = torch.stack(trajectory[t : t + window_size + 1], dim=0)
            updated_states = torch.stack(updated, dim=0)
            stride, error_ratios = _picard_stride(
                previous,
                updated_states,
                t,
                window_size,
                num_steps,
                tol,
                threshold_action_steps,
            )

            for index in range(1, window_size + 1):
                trajectory[t + index] = updated[index]
            v_free = velocities[stride - 1].detach()
            rounds.append(
                {
                    "t": int(t),
                    "window": int(window_size),
                    "stride": int(stride),
                    "max_error_ratio": float(max(error_ratios, default=0.0)),
                }
            )
            t += stride
            stride_sum += stride

        self.last_repic_trajectory = torch.stack(trajectory[: num_steps + 1], dim=0).detach().cpu()
        self.last_repic_stats = {
            "nfe": int(nfe),
            "num_rounds": len(rounds),
            "num_steps": num_steps,
            "picard_window": window,
            "picard_tol": tol,
            "mean_stride": float(stride_sum) / max(len(rounds), 1),
            "init_mode": init_mode,
            "warm_start_replan_steps": replan_steps,
            "threshold_action_steps": threshold_action_steps,
            "rounds": rounds,
        }
        return BatchFeature(data={"action_pred": trajectory[num_steps]})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
