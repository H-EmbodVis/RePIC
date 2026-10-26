"""CPU tests for the RoboTwin RePIC policy logic (replan schedule, cache reset, kwargs).

The dataset/processor modules need ``lerobot``, ``decord`` and friends, which are irrelevant here, so they
are stubbed before the policy module is imported.
"""
import importlib.util
import json
import math
import sys
import types
from collections import deque
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

ROOT = Path(__file__).resolve().parents[1]
H = 32


@pytest.fixture(scope="module")
def policy_module():
    stubs = {
        "fastwam.datasets.lerobot.processors.fastwam_processor": {"FastWAMProcessor": object},
        "fastwam.datasets.lerobot.robot_video_dataset": {"DEFAULT_PROMPT": "{task}"},
        "fastwam.datasets.lerobot.utils.normalizer": {"load_dataset_stats_from_json": lambda *_: None},
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
    try:
        path = ROOT / "experiments" / "robotwin" / "fastwam_policy" / "deploy_policy.py"
        spec = importlib.util.spec_from_file_location("repic_robotwin_deploy_policy", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class _FakeModel:
    """Records the kwargs of ``infer_action`` and returns a RePIC-style result."""

    device = "cpu"
    torch_dtype = torch.float32

    def __init__(self, residual_value=1.0):
        self.calls = []
        self.residual_value = residual_value

    def infer_action(self, **kwargs):
        self.calls.append(kwargs)
        steps = kwargs["num_inference_steps"]
        horizon = kwargs["action_horizon"]
        return {
            "action": torch.zeros(horizon, 4),
            "picard_stats": {
                "final_trajectory": torch.zeros(steps + 1, horizon, 4),
                "final_clean_action_residual": torch.full((horizon,), self.residual_value),
            },
        }


def _make_policy(module, name="robotwin", *, repic=True, action_horizon=H, replan_steps=None):
    config = json.loads((ROOT / "configs" / "repic" / f"{name}.json").read_text())
    policy = object.__new__(module.WorldActionRobotWinPolicy)
    policy.action_horizon = action_horizon
    policy.replan_steps = replan_steps or config["r_max"]
    policy.num_inference_steps = 20
    policy.sigma_shift = None
    policy.seed = None
    policy.text_cfg_scale = 1.0
    policy.negative_prompt = ""
    policy.rand_device = "cpu"
    policy.tiled = False
    policy.timing_enabled = False
    policy._num_video_frames = 9
    policy.repic = repic
    policy.repic_flash = False
    policy.repic_config = config
    policy.picard_trajectory = None
    policy.pending_actions = deque()
    policy.episode_count = 0
    policy.step_count = 0
    policy._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
    policy._initial_replan_steps = policy.replan_steps
    policy._validate_repic_config()
    policy._reset_replan_state()
    policy.model = _FakeModel()
    policy._build_robotwin_image_tensor = lambda obs: torch.zeros(1, 3, 16, 16)
    policy._normalize_state = lambda state: torch.zeros(1, 4)
    policy._denormalize_action = lambda action: action.unsqueeze(0).numpy()
    return policy


OBS = {"joint_action": {"vector": [0.0, 0.0, 0.0, 0.0]}}


def test_validate_config_derives_paper_quantities(policy_module):
    policy = _make_policy(policy_module)
    assert policy.num_inference_steps == 10  # overridden by the RePIC config
    assert policy.r_max == 24 and policy.r_min == 16  # R_min = ceil(H/2)
    assert policy.dynamic_replan_active
    assert policy.stride_residual == "prefix"
    assert policy._active_replan_steps == policy._prev_replan_steps == 24

    libero = _make_policy(policy_module, "libero")
    assert libero.r_max == 10 and not libero.dynamic_replan_active


def test_validate_config_rejects_bad_values(policy_module):
    policy = _make_policy(policy_module)
    policy.repic_config = {**policy.repic_config, "stride_residual": "tail"}
    with pytest.raises(ValueError):
        policy._validate_repic_config()
    policy.repic_config = {**policy.repic_config, "stride_residual": "prefix", "r_max": H + 1}
    with pytest.raises(ValueError):
        policy._validate_repic_config()


@pytest.mark.parametrize(
    "u, expected",
    [(1e-8, 24), (math.exp(-3.5), 20), (1e3, 16)],  # Phi(z) -> 0, 0.5, 1
)
def test_commitment_map_matches_eq_9(policy_module, u, expected):
    policy = _make_policy(policy_module)
    residual = torch.full((H,), u)
    assert policy._select_next_replan_steps(residual, executed_steps=24) == expected


def test_commitment_map_is_monotone_and_bounded(policy_module):
    policy = _make_policy(policy_module)
    picks = [policy._select_next_replan_steps(torch.full((H,), u), 24) for u in torch.logspace(-6, 2, 40).tolist()]
    assert all(policy.r_min <= p <= policy.r_max for p in picks)
    assert picks == sorted(picks, reverse=True)  # larger residual -> shorter prefix


def test_commitment_uses_only_the_unexecuted_tail(policy_module):
    policy = _make_policy(policy_module)
    residual = torch.cat([torch.full((24,), 1e3), torch.full((H - 24,), 1e-8)])
    assert policy._select_next_replan_steps(residual, executed_steps=24) == 24  # executed prefix is ignored
    assert policy._select_next_replan_steps(residual, executed_steps=8) == 16


def test_commitment_requires_sampler_residual(policy_module):
    with pytest.raises(RuntimeError):
        _make_policy(policy_module)._select_next_replan_steps(None, 24)


def test_first_chunk_is_cold_and_second_chunk_is_warm_with_delayed_commitment(policy_module):
    policy = _make_policy(policy_module)
    policy.model.residual_value = 1e3  # large tail residual -> next prefix is R_min
    policy._infer_action_chunk(OBS, "task")
    first = policy.model.calls[0]
    assert first["picard_sampling"] and first["picard_threshold_action_steps"] == 24
    assert "picard_init_trajectory" not in first  # cold start
    assert policy._chunk_replan_steps == 24 and policy._prev_replan_steps == 24
    assert policy._active_replan_steps == 16  # R_2 chosen from chunk 1's tail (one-replan delay)

    policy._infer_action_chunk(OBS, "task")
    second = policy.model.calls[1]
    assert second["picard_threshold_action_steps"] == 16  # Eq. 4 on 1:R_2
    assert second["picard_warm_start_replan_steps"] == 24  # Eq. 6 shift by R_1
    assert second["picard_init_trajectory"].shape == (policy.num_inference_steps + 1, H, 4)
    assert policy._prev_replan_steps == 16


def test_actions_executed_per_chunk_follow_the_committed_length(policy_module):
    policy = _make_policy(policy_module)
    policy.model.residual_value = 1e3
    policy._fill_action_queue(OBS, "task")
    assert len(policy.pending_actions) == 24
    policy.pending_actions.clear()
    policy._fill_action_queue(OBS, "task")
    assert len(policy.pending_actions) == 16


def test_reset_starts_the_next_episode_cold(policy_module):
    policy = _make_policy(policy_module)
    policy.model.residual_value = 1e3
    policy._fill_action_queue(OBS, "task")
    assert policy.picard_trajectory is not None and policy._active_replan_steps == 16
    policy.reset()
    assert policy.picard_trajectory is None
    assert policy._active_replan_steps == policy._prev_replan_steps == policy._chunk_replan_steps == 24
    assert not policy.pending_actions
    policy.model.calls.clear()
    policy._infer_action_chunk(OBS, "task")
    assert "picard_init_trajectory" not in policy.model.calls[0]


def test_fixed_replan_on_libero_style_config_never_changes(policy_module):
    policy = _make_policy(policy_module, "libero")
    policy.model.residual_value = 1e3
    for _ in range(3):
        policy._fill_action_queue(OBS, "task")
        assert policy._chunk_replan_steps == 10
        policy.pending_actions.clear()
    assert policy.model.calls[-1]["picard_warm_start_replan_steps"] == 10


def test_full_stride_residual_does_not_pass_a_prefix(policy_module):
    policy = _make_policy(policy_module)
    policy.repic_config = {**policy.repic_config, "stride_residual": "full"}
    policy._validate_repic_config()
    policy._infer_action_chunk(OBS, "task")
    assert "picard_threshold_action_steps" not in policy.model.calls[0]


def test_baseline_mode_does_not_touch_repic_state(policy_module):
    policy = _make_policy(policy_module, repic=False)
    policy._fill_action_queue(OBS, "task")
    call = policy.model.calls[0]
    assert "picard_sampling" not in call and "repic_flash" not in call
    assert len(policy.pending_actions) == policy.replan_steps
