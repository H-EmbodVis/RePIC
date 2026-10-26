import json
import types
from pathlib import Path

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    _noise_aligned_warm_start,
    _picard_stride,
)


def test_libero_repic_config_matches_paper_setting():
    config = json.loads(Path("configs/repic/libero.json").read_text())
    assert config["action_horizon"] == 16
    assert config["num_inference_steps"] == 4
    assert config["picard_window"] == 4
    assert config["picard_tol"] == 0.1
    assert config["r_max"] == 1
    assert config["r_min"] == 8
    assert config["dynamic_replan_enabled"] is False


def test_noise_aligned_warm_start_shifts_unexecuted_tail():
    cached = torch.arange(5 * 1 * 4, dtype=torch.float32).reshape(5, 1, 4, 1)
    noise = torch.full((1, 4, 1), -7.0)
    warm = _noise_aligned_warm_start(cached, noise, replan_steps=1)
    torch.testing.assert_close(warm[0], noise)
    torch.testing.assert_close(warm[-1, :, :3], cached[-1, :, 1:])


def test_picard_stride_uses_prefix_residual_and_noise_scaled_threshold():
    previous = torch.zeros((4, 1, 2, 1))
    updated = previous.clone()
    updated[1, :, 0, 0] = 0.2
    stride, ratios = _picard_stride(
        previous,
        updated,
        start_step=0,
        window_size=3,
        num_steps=4,
        tolerance=0.1,
        action_steps=1,
    )
    assert stride == 1
    assert ratios[0] > 1.0


def test_picard_sampler_reaches_clean_endpoint_with_fake_head():
    class FakeHead:
        action_horizon = 2
        action_dim = 1
        num_inference_timesteps = 4
        num_timestep_buckets = 1000
        config = types.SimpleNamespace(add_pos_embed=False)
        process_backbone_output = staticmethod(lambda value: value)
        state_encoder = staticmethod(
            lambda state, embodiment: torch.zeros((state.shape[0], 1, 1), dtype=state.dtype)
        )
        action_encoder = staticmethod(lambda actions, timesteps, embodiment: actions)
        future_tokens = types.SimpleNamespace(weight=torch.zeros((1, 1)))
        model = staticmethod(
            lambda hidden_states, encoder_hidden_states, encoder_attention_mask, timestep: hidden_states
        )
        action_decoder = staticmethod(lambda model_output, embodiment: model_output[..., :1])

        def __init__(self):
            self.last_repic_trajectory = None
            self.last_repic_stats = None

    head = FakeHead()
    backbone = BatchFeature(
        data={"backbone_features": torch.zeros((1, 1, 1)), "backbone_attention_mask": None}
    )
    action_input = BatchFeature(
        data={
            "state": torch.zeros((1, 1, 1)),
            "embodiment_id": torch.zeros((1,), dtype=torch.long),
        }
    )
    result = FlowmatchingActionHead.get_action_repic(
        head, backbone, action_input, picard_window=4, picard_tol=0.1
    )
    assert tuple(result["action_pred"].shape) == (1, 2, 1)
    assert tuple(head.last_repic_trajectory.shape) == (5, 1, 2, 1)
    assert head.last_repic_stats["nfe"] >= head.last_repic_stats["num_rounds"]
