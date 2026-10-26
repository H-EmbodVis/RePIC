import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ACTION_HORIZON = 32  # num_frames - 1 for both LIBERO and RoboTwin (configs/data/*.yaml)


def _load(name):
    return json.loads((ROOT / "configs" / "repic" / f"{name}.json").read_text())


@pytest.mark.parametrize("name, r_max", [("libero", 10), ("robotwin", 24)])
def test_repic_configs_follow_paper_defaults(name, r_max):
    config = _load(name)
    assert config["num_inference_steps"] == 10  # S
    assert config["picard_window"] == 10  # p
    assert config["picard_tol"] == 0.05  # tau
    assert config["r_max"] == r_max
    assert config["stride_residual"] == "prefix"  # Eq. 4 on 1:R_n
    assert config["dynamic_replan_log_mean"] == -3.5  # mu
    assert config["dynamic_replan_log_std"] == 0.5  # varsigma


def test_commitment_is_enabled_only_where_it_can_act():
    # Paper App. A: R_min = ceil(H/2); commitment is disabled whenever R_max <= R_min.
    r_min = math.ceil(ACTION_HORIZON / 2)
    assert r_min == 16
    for name in ("libero", "robotwin"):
        config = _load(name)
        assert config["dynamic_replan_enabled"] == (config["r_max"] > r_min)
        assert config["r_max"] <= ACTION_HORIZON
