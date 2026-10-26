import time
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        compile_training_denoise: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.compile_training_denoise = bool(compile_training_denoise)
        self.mot.compile_training_layers = self.compile_training_denoise
        self._action_cuda_graph_enabled = False
        self._action_cuda_graphs: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._action_cuda_graph_stats = {"requests": 0, "captures": 0, "replays": 0}
        self._action_tensorrt_enabled = False
        self._action_tensorrt_mixed_precision = True
        self._action_tensorrt_precision = torch.float16
        self._action_tensorrt_engine_batch = 10
        self._action_tensorrt_engines: dict[int, Any] = {}
        self._action_tensorrt_core = None

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        compile_training_denoise: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            compile_training_denoise=compile_training_denoise,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if tiled:
            raise NotImplementedError("Batched VAE encoding does not support tiled encoding.")
        if not hasattr(self, "_vae_encode_compiled"):
            self._vae_encode_compiled = torch.compile(
                self.vae.model.encode,
                backend="cudagraphs",
                fullgraph=True,
            )
        return self._vae_encode_compiled(
            video_tensor.to(self.device),
            self.vae.scale,
        ).clone()

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        if tiled:
            raise NotImplementedError("Batched VAE image encoding does not support tiled encoding.")
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        return self.vae.model.encode(image.unsqueeze(0), self.vae.scale)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None and context_mask is None:
            prompt = sample.get("prompt")
            if prompt is None:
                raise ValueError("FastWAM training requires `context/context_mask` or `prompt`.")
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must both exist when either is provided.")

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _joint_denoise_core(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the tensor-only video/action core shared by training and inference."""
        (
            video_tokens,
            t_video,
            t_mod_video,
            context_video,
            context_mask_video,
            freqs_video,
            f_video,
            h_video,
            w_video,
            _tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action_condition,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        video_tokens, action_tokens = self.mot.forward_joint_core(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_freqs=freqs_video,
            action_freqs=freqs_action,
            video_t_mod=t_mod_video,
            action_t_mod=t_mod_action,
            video_context=context_video,
            video_context_mask=context_mask_video,
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=attention_mask,
        )
        return (
            self.video_expert.post(video_tokens, t_video, f_video, h_video, w_video),
            self.action_expert.post(action_tokens),
        )

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = latents.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=noisy_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents.device,
        )
        pred_video, pred_action = self._joint_denoise_core(
            latents_video=latents,
            latents_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            action_condition=action,
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = latents_video.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents_video.device,
        )
        return self._joint_denoise_core(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            action_condition=gt_action,
        )

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    def _denoise_action_with_video_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        (
            action_tokens,
            _t,
            action_t_mod,
            action_context,
            action_context_mask,
            action_freqs,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache_tensor(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context=action_context,
            action_context_mask=action_context_mask,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask,
        )
        return self.action_expert.post(action_tokens)

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Legacy dictionary-cache path retained for the optional IDM variant."""
        batch_size = int(latents_action.shape[0])
        if context.shape[0] == 1 and batch_size > 1:
            context = context.expand(batch_size, *context.shape[1:])
            context_mask = context_mask.expand(batch_size, *context_mask.shape[1:])
            video_kv_cache = [
                {"k": layer["k"].expand(batch_size, *layer["k"].shape[1:]),
                 "v": layer["v"].expand(batch_size, *layer["v"].shape[1:])}
                for layer in video_kv_cache
            ]
        if self._action_cuda_graph_enabled:
            if not latents_action.is_cuda:
                raise RuntimeError("RePIC-Flash CUDA Graphs require CUDA action tensors.")
            return self._predict_action_noise_with_cuda_graph(
                latents_action=latents_action, timestep_action=timestep_action,
                context=context, context_mask=context_mask, video_kv_cache=video_kv_cache,
                attention_mask=attention_mask, video_seq_len=video_seq_len,
            )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def set_action_cuda_graph(self, enabled: bool) -> None:
        if enabled and not torch.cuda.is_available():
            raise RuntimeError("RePIC-Flash LIBERO backend requires CUDA.")
        enabled = bool(enabled)
        if enabled == self._action_cuda_graph_enabled:
            # `infer_action` re-asserts the backend on every replan. Keep the graphs that
            # are already captured: each shape is captured once and replayed afterwards.
            return
        self._action_cuda_graph_enabled = enabled
        self._action_cuda_graphs.clear()
        self._action_cuda_graph_stats = {"requests": 0, "captures": 0, "replays": 0}

    def set_action_tensorrt(self, enabled: bool, **_: Any) -> None:
        if enabled:
            if not torch.cuda.is_available():
                raise RuntimeError("RePIC-Flash RoboTwin backend requires CUDA.")
            try:
                import torch_tensorrt  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("RePIC-Flash RoboTwin backend requires torch_tensorrt.") from exc
        self._action_tensorrt_enabled = bool(enabled)
        if not enabled:
            self._action_tensorrt_engines.clear()
            self._action_tensorrt_core = None

    def _predict_action_noise_with_cuda_graph(self, **kwargs: Any) -> torch.Tensor:
        latents_action = kwargs["latents_action"]
        timestep_action = kwargs["timestep_action"]
        context = kwargs["context"]
        context_mask = kwargs["context_mask"]
        video_kv_cache = kwargs["video_kv_cache"]
        attention_mask = kwargs["attention_mask"]
        video_seq_len = kwargs["video_seq_len"]
        if self._action_tensorrt_enabled:
            shape = (self._action_tensorrt_engine_batch, *latents_action.shape[1:])
            ctx_shape = (self._action_tensorrt_engine_batch, *context.shape[1:])
            cache_shape = tuple(((self._action_tensorrt_engine_batch, *x["k"].shape[1:]), (self._action_tensorrt_engine_batch, *x["v"].shape[1:])) for x in video_kv_cache)
        else:
            shape, ctx_shape = tuple(latents_action.shape), tuple(context.shape)
            cache_shape = tuple((tuple(x["k"].shape), tuple(x["v"].shape)) for x in video_kv_cache)
        key = (shape, ctx_shape, tuple(attention_mask.shape), video_seq_len, cache_shape, str(latents_action.dtype), self._action_tensorrt_enabled)
        state = self._action_cuda_graphs.get(key)
        if state is None:
            state = self._create_action_cuda_graph(**kwargs)
            self._action_cuda_graphs[key] = state
            self._action_cuda_graph_stats["captures"] += 1
        def copy_source(dst, src):
            if not state.get("tensorrt_padded_batch", False):
                dst.copy_(src)
                return
            dst.copy_(src[:1].expand_as(dst))
            dst[:src.shape[0]].copy_(src)
        copy_source(state["latents"], latents_action)
        copy_source(state["timesteps"], timestep_action)
        copy_source(state["context"], context)
        copy_source(state["context_mask"], context_mask)
        state["attention_mask"].copy_(attention_mask)
        for dst, src in zip(state["video_kv_cache"], video_kv_cache):
            copy_source(dst["k"], src["k"])
            copy_source(dst["v"], src["v"])
        state["graph"].replay()
        self._action_cuda_graph_stats["requests"] += 1
        self._action_cuda_graph_stats["replays"] += 1
        output = state["output"]
        return output[:latents_action.shape[0]].clone() if state.get("tensorrt_padded_batch", False) else output.clone()

    def _create_action_cuda_graph(self, **kwargs: Any) -> dict[str, Any]:
        latents_action = kwargs["latents_action"]
        timestep_action = kwargs["timestep_action"]
        context = kwargs["context"]
        context_mask = kwargs["context_mask"]
        video_kv_cache = kwargs["video_kv_cache"]
        attention_mask = kwargs["attention_mask"]
        video_seq_len = kwargs["video_seq_len"]
        if self._action_tensorrt_enabled:
            return self._create_tensorrt_action_cuda_graph(**kwargs)
        static = {
            "latents": torch.empty_like(latents_action), "timesteps": torch.empty_like(timestep_action),
            "context": torch.empty_like(context), "context_mask": torch.empty_like(context_mask),
            "attention_mask": torch.empty_like(attention_mask),
            "video_kv_cache": [{"k": torch.empty_like(x["k"]), "v": torch.empty_like(x["v"])} for x in video_kv_cache],
        }
        static["latents"].copy_(latents_action); static["timesteps"].copy_(timestep_action)
        static["context"].copy_(context); static["context_mask"].copy_(context_mask); static["attention_mask"].copy_(attention_mask)
        for dst, src in zip(static["video_kv_cache"], video_kv_cache):
            dst["k"].copy_(src["k"]); dst["v"].copy_(src["v"])
        def run() -> torch.Tensor:
            action_pre = self.action_expert.pre_dit(static["latents"], static["timesteps"], static["context"], static["context_mask"])
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"], action_freqs=action_pre["freqs"], action_t_mod=action_pre["t_mod"],
                action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                video_kv_cache=static["video_kv_cache"], attention_mask=static["attention_mask"], video_seq_len=video_seq_len)
            return self.action_expert.post_dit(action_tokens, action_pre)
        try:
            for _ in range(3):
                output = run()
            static["output"] = torch.empty_like(output)
            torch.cuda.synchronize(latents_action.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static["output"].copy_(run())
            torch.cuda.synchronize(latents_action.device)
        except Exception as exc:
            raise RuntimeError("RePIC-Flash CUDA Graph capture failed; no eager fallback is allowed.") from exc
        static["graph"] = graph
        return static

    def _create_tensorrt_action_cuda_graph(self, **kwargs: Any) -> dict[str, Any]:
        import torch_tensorrt

        latents_action = kwargs["latents_action"]
        timestep_action = kwargs["timestep_action"]
        context = kwargs["context"]
        context_mask = kwargs["context_mask"]
        video_kv_cache = kwargs["video_kv_cache"]
        attention_mask = kwargs["attention_mask"]
        video_seq_len = int(kwargs["video_seq_len"])
        engine_batch = int(self._action_tensorrt_engine_batch)
        if int(latents_action.shape[0]) > engine_batch:
            raise ValueError(f"TensorRT action batch {latents_action.shape[0]} exceeds {engine_batch}.")
        def full_batch(source):
            return torch.empty((engine_batch, *source.shape[1:]), dtype=source.dtype, device=source.device)
        def copy_full(dst, src):
            dst.copy_(src[:1].expand_as(dst)); dst[:src.shape[0]].copy_(src)
        static_latents, static_timesteps = full_batch(latents_action), full_batch(timestep_action)
        static_context, static_context_mask = full_batch(context), full_batch(context_mask)
        static_attention_mask = torch.empty_like(attention_mask)
        static_cache = [{"k": full_batch(layer["k"].to(dtype=torch.float16)), "v": full_batch(layer["v"].to(dtype=torch.float16))} for layer in video_kv_cache]
        copy_full(static_latents, latents_action); copy_full(static_timesteps, timestep_action)
        copy_full(static_context, context); copy_full(static_context_mask, context_mask); static_attention_mask.copy_(attention_mask)
        for dst, src in zip(static_cache, video_kv_cache):
            copy_full(dst["k"], src["k"]); copy_full(dst["v"], src["v"])
        engine = self._action_tensorrt_engines.get(engine_batch)
        try:
            if engine is None:
                from .tensorrt_action import TensorRTActionCore
                if self._action_tensorrt_core is None:
                    object.__setattr__(self, "_action_tensorrt_core", TensorRTActionCore(self.action_expert, weight_dtype=torch.float16).to(self.device).eval())
                pre = self.action_expert.pre_dit(static_latents, static_timesteps, static_context, static_context_mask)
                inputs = (pre["tokens"].half(), pre["t_mod"].half(), pre["context"].half(), pre["context_mask"], static_attention_mask[video_seq_len:, :], *(x for layer in static_cache for x in (layer["k"], layer["v"])))
                traced = torch.jit.trace(self._action_tensorrt_core, inputs, check_trace=False, strict=False)
                engine = torch_tensorrt.compile(traced, ir="ts", inputs=list(inputs), enabled_precisions={torch.float16}, require_full_compilation=True, workspace_size=8 * 1024**3)
                self._action_tensorrt_engines[engine_batch] = engine
        except Exception as exc:
            raise RuntimeError("TensorRT FP16 action engine construction failed; no eager fallback is allowed.") from exc
        def run_static():
            pre = self.action_expert.pre_dit(static_latents, static_timesteps, static_context, static_context_mask)
            return engine(pre["tokens"].half(), pre["t_mod"].half(), pre["context"].half(), pre["context_mask"], static_attention_mask[video_seq_len:, :], *(x for layer in static_cache for x in (layer["k"], layer["v"]))).to(dtype=static_latents.dtype)
        for _ in range(3):
            output = run_static()
        static_output = torch.empty_like(output)
        torch.cuda.synchronize(latents_action.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output.copy_(run_static())
        torch.cuda.synchronize(latents_action.device)
        return {"graph": graph, "latents": static_latents, "timesteps": static_timesteps, "context": static_context, "context_mask": static_context_mask, "attention_mask": static_attention_mask, "video_kv_cache": static_cache, "output": static_output, "tensorrt_padded_batch": True}

    @staticmethod
    def _fm_trace_ta(x: torch.Tensor) -> torch.Tensor:
        y = x.detach()
        if y.ndim == 3:
            if int(y.shape[0]) != 1:
                raise ValueError(f"Expected batch size 1, got {tuple(y.shape)}")
            y = y[0]
        if y.ndim != 2:
            raise ValueError(f"Expected [T,A] or [1,T,A], got {tuple(y.shape)}")
        return y.to(device="cpu", dtype=torch.float32)

    @torch.no_grad()
    def _infer_action_picard_loop(
        self, *, latents_action, infer_timesteps_action, infer_deltas_action,
        context, context_mask, video_kv_cache, attention_mask, video_seq_len,
        window: int, tol: float, init_trajectory: Optional[torch.Tensor] = None,
        warm_start_replan_steps: Optional[int] = None,
        threshold_action_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sliding-window Picard sampler of RePIC (Eq. 3-5, Algorithm 1).

        ``threshold_action_steps`` is the executed prefix length ``R_n``: the stride test of
        Eq. 4 is evaluated on the first ``R_n`` action positions only (paper default). ``None``
        evaluates it on the whole chunk (the ``Full 1:H`` ablation of Tab. 5). All ``H``
        positions are always updated and written back.
        """
        sched = self.infer_action_scheduler
        steps = int(infer_timesteps_action.numel())
        if steps <= 0 or window <= 0 or tol < 0:
            raise ValueError("Picard steps, window, and tolerance must be valid")
        if init_trajectory is not None:
            if warm_start_replan_steps is None:
                raise ValueError("`picard_init_trajectory` requires `picard_warm_start_replan_steps`")
            warm, _ = self._picard_noise_aligned_warm_trajectory(
                init_trajectory=init_trajectory, infer_timesteps_action=infer_timesteps_action,
                num_train_timesteps=int(sched.num_train_timesteps), num_steps=steps,
                ref=latents_action, replan_steps=int(warm_start_replan_steps))
            xs = [warm[i:i + 1].contiguous() for i in range(steps + 1)]
            filled = steps
            init_mode = "noise_aligned_history"
        else:
            xs = [latents_action]
            filled = 0
            init_mode = "linear"
        t = 0
        nfe = 0
        rounds = []
        v_free = None
        final_residual = None
        horizon = int(latents_action.shape[1])
        if threshold_action_steps is None:
            threshold_steps = horizon
        else:
            threshold_steps = int(threshold_action_steps)
            if not 1 <= threshold_steps <= horizon:
                raise ValueError("`picard_threshold_action_steps` must be in [1, action_horizon]")
        dim = threshold_steps * int(latents_action.shape[2])
        while t < steps:
            p = min(int(window), steps - t)
            while len(xs) < t + p + 1:
                xs.append(xs[-1].clone())
            if filled < t + p:
                if v_free is None:
                    v_draft = self._predict_action_noise_with_cache(
                        latents_action=xs[t], timestep_action=infer_timesteps_action[t:t + 1],
                        context=context, context_mask=context_mask, video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask, video_seq_len=video_seq_len)
                    nfe += 1
                else:
                    v_draft = v_free
                draft = sched.draft_roll_frozen(v_draft, infer_deltas_action[filled:t + p], xs[filled])
                for j in range(1, len(draft)):
                    xs[filled + j] = draft[j]
                filled = t + p
            batch = torch.cat(xs[t:t + p], dim=0)
            velocities = self._predict_action_noise_with_cache(
                latents_action=batch, timestep_action=infer_timesteps_action[t:t + p],
                context=context, context_mask=context_mask, video_kv_cache=video_kv_cache,
                attention_mask=attention_mask, video_seq_len=video_seq_len)
            nfe += 1
            updated = sched.picard_prefix_integrate(velocities, infer_deltas_action[t:t + p], xs[t])
            if t + p == steps:
                final_residual = (updated[-1] - xs[t + p])[0].float().pow(2).mean(dim=-1).sqrt().detach().cpu()
            stride = p
            ratios = []
            for j in range(1, p):
                diff = (updated[j] - xs[t + j])[..., :threshold_steps, :].float()
                error = float(diff.pow(2).sum().item()) / float(dim)
                threshold = float(tol * tol * sched.forward_process_variance(infer_timesteps_action[t + j]).item())
                ratio = error / threshold if threshold > 0 else float("inf")
                ratios.append(ratio)
                if stride == p and error > threshold:
                    stride = j
            for j in range(1, p + 1):
                xs[t + j] = updated[j]
            v_free = velocities[stride - 1:stride].detach()
            rounds.append({"t": t, "p": p, "stride": stride, "max_error_ratio": max(ratios, default=0.0)})
            t += stride
        final = torch.stack([self._fm_trace_ta(x) for x in xs], dim=0)
        return xs[-1], {"nfe": nfe, "num_rounds": len(rounds), "mean_stride": sum(r["stride"] for r in rounds) / max(len(rounds), 1), "picard_window": window, "picard_tol": tol, "init_mode": init_mode, "rounds": rounds, "final_clean_action_residual": final_residual, "final_trajectory": final, "picard_threshold_action_steps": threshold_steps}

    @staticmethod
    def _picard_noise_aligned_warm_trajectory(
        *, init_trajectory: torch.Tensor, infer_timesteps_action: torch.Tensor,
        num_train_timesteps: int, num_steps: int, ref: torch.Tensor,
        replan_steps: int, mode: str = "full_path_shift_align",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        aliases = {"shift_align": "full_path_shift_align", "noise_aligned_history": "full_path_shift_align"}
        mode = aliases.get(str(mode).strip().lower(), str(mode).strip().lower())
        if mode != "full_path_shift_align":
            raise ValueError("Only the paper's full_path_shift_align warm start is supported.")
        traj = init_trajectory.detach()
        if traj.ndim == 4 and traj.shape[1] == 1:
            traj = traj[:, 0]
        if tuple(traj.shape) != (num_steps + 1, *ref.shape[1:]):
            raise ValueError("`picard_init_trajectory` has an incompatible shape")
        horizon = int(ref.shape[1])
        if not 0 < int(replan_steps) < horizon:
            raise ValueError("`picard_warm_start_replan_steps` must be in [1, action_horizon)")
        tail = horizon - int(replan_steps)
        indices = torch.arange(horizon, device=ref.device)
        indices[:tail] += int(replan_steps)
        source = traj.to(device=ref.device, dtype=ref.dtype).index_select(1, indices)
        sigma = torch.cat((infer_timesteps_action, infer_timesteps_action.new_zeros(1)))
        sigma = (sigma / float(num_train_timesteps)).to(dtype=ref.dtype).view(-1, 1, 1)
        warm = source + sigma * (ref[0].unsqueeze(0) - source[0:1])
        return warm.contiguous(), {"mode": "noise_aligned_history", "tail_actions": tail, "direct_actions": int(replan_steps), "warm_actions": horizon, "scene_nfe": 0}

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        compile_action_infer: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                compile_action_infer=compile_action_infer,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        joint_attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        if compile_action_infer:
            if action is not None:
                raise ValueError(
                    "`compile_action_infer=True` does not support action conditioning in `infer_joint`."
                )
            if not hasattr(self, "_joint_denoise_core_compiled_inference"):
                self._joint_denoise_core_compiled_inference = torch.compile(
                    self._joint_denoise_core,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            joint_denoise_core = self._joint_denoise_core_compiled_inference
        else:
            joint_denoise_core = self._joint_denoise_core

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            if compile_action_infer:
                torch.compiler.cudagraph_mark_step_begin()
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = joint_denoise_core(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                attention_mask=joint_attention_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                action_condition=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        picard_sampling: bool = False,
        picard_window: int = 10,
        picard_tol: float = 0.05,
        picard_init_trajectory: Optional[torch.Tensor] = None,
        picard_warm_start_replan_steps: Optional[int] = None,
        picard_threshold_action_steps: Optional[int] = None,
        repic_flash: bool = False,
        repic_flash_backend: str = "cuda_graph",
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        if repic_flash and not torch.cuda.is_available():
            raise RuntimeError("RePIC-Flash requires CUDA; eager fallback is disabled.")

        if picard_sampling or repic_flash:
            timestep_video = torch.zeros(
                (first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=self.device
            )
            video_pre = self.video_expert.pre_dit(
                x=first_frame_latents, timestep=timestep_video, context=context,
                context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len, action_seq_len=latents_action.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]), device=self.device,
            )
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"], video_freqs=video_pre["freqs"], video_t_mod=video_pre["t_mod"],
                video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
            infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps, device=self.device, dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
            if repic_flash:
                backend = str(repic_flash_backend).strip().lower()
                if backend not in {"cuda_graph", "tensorrt"}:
                    raise ValueError("`repic_flash_backend` must be `cuda_graph` or `tensorrt`.")
                self.set_action_cuda_graph(True)
                if backend == "tensorrt":
                    self.set_action_tensorrt(True)
            if repic_flash and not picard_sampling:
                for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
                    timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                    pred_action = self._predict_action_noise_with_cache(
                        latents_action=latents_action, timestep_action=timestep_action,
                        context=context, context_mask=context_mask, video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask, video_seq_len=video_seq_len)
                    latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
                return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
            latents_action, picard_stats = self._infer_action_picard_loop(
                latents_action=latents_action, infer_timesteps_action=infer_timesteps_action,
                infer_deltas_action=infer_deltas_action, context=context, context_mask=context_mask,
                video_kv_cache=video_kv_cache, attention_mask=attention_mask, video_seq_len=video_seq_len,
                window=int(picard_window), tol=float(picard_tol), init_trajectory=picard_init_trajectory,
                warm_start_replan_steps=picard_warm_start_replan_steps,
                threshold_action_steps=picard_threshold_action_steps,
            )
            return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32), "picard_stats": picard_stats}

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f_video,
            _h_video,
            _w_video,
            tokens_per_frame,
        ) = self.video_expert.prepare(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_tokens.shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        video_attention_mask = attention_mask[:video_seq_len, :video_seq_len]
        action_attention_mask = attention_mask[video_seq_len:, :]
        if compile_action_infer:
            if not hasattr(self, "_prefill_video_cache_compiled"):
                self._prefill_video_cache_compiled = torch.compile(
                    self.mot.prefill_video_cache_tensor,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            if not hasattr(self, "_denoise_action_with_video_cache_compiled"):
                self._denoise_action_with_video_cache_compiled = torch.compile(
                    self._denoise_action_with_video_cache,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            prefill_video_cache = self._prefill_video_cache_compiled
            denoise_action_with_video_cache = self._denoise_action_with_video_cache_compiled
        else:
            prefill_video_cache = self.mot.prefill_video_cache_tensor
            denoise_action_with_video_cache = self._denoise_action_with_video_cache
        if compile_action_infer:
            torch.compiler.cudagraph_mark_step_begin()
        video_cache_k, video_cache_v = prefill_video_cache(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        if compile_action_infer:
            # Inductor reduce-overhead may return graph-owned buffers that are overwritten on replay.
            video_cache_k = [cache.clone() for cache in video_cache_k]
            video_cache_v = [cache.clone() for cache in video_cache_v]

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            if compile_action_infer:
                torch.compiler.cudagraph_mark_step_begin()
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = denoise_action_with_video_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_cache_k=video_cache_k,
                video_cache_v=video_cache_v,
                action_attention_mask=action_attention_mask,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
