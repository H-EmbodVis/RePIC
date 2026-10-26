"""CPU unit tests for the RePIC sampler (Eqs. 3-7 of the paper).

The velocity network is replaced by an analytic, position-wise field, so the tests need
neither checkpoints nor a GPU. They check that

* Picard sampling reproduces plain serial Euler sampling (tolerance 0) and collapses to a
  single batched call when the draft is already exact (Eq. 3-5);
* the noise-aligned transfer of Eqs. 6-7 has the properties claimed in the paper;
* the stride test of Eq. 4 is evaluated on the executed prefix ``1:R_n`` only, while the
  ``Full 1:H`` ablation uses the whole chunk.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.wan22.fastwam import FastWAM  # noqa: E402
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler  # noqa: E402

H, A, STEPS, SHIFT = 32, 7, 10, 5.0


def _velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Smooth position-wise velocity field v(x, t); batch element ``b`` uses ``t[b]``."""
    s = (t.to(x.dtype) / 1000.0).view(-1, 1, 1)
    return -0.8 * x + 0.5 * torch.sin(3.0 * x) * (1.0 - s) + 0.3 * torch.cos(2.0 * s)


def _constant_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.full_like(x, 0.25)


class _Sampler(FastWAM):
    """FastWAM with the action network replaced by an analytic velocity field."""

    def __init__(self, field):  # noqa: D107 - deliberately skips the heavy FastWAM constructor
        torch.nn.Module.__init__(self)
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(1000, SHIFT)
        self.field = field
        self.calls = 0

    def _predict_action_noise_with_cache(self, *, latents_action, timestep_action, **_):
        self.calls += 1
        assert latents_action.shape[0] == timestep_action.shape[0]
        return self.field(latents_action, timestep_action)


@pytest.fixture
def schedule():
    sched = WanContinuousFlowMatchScheduler(1000, SHIFT)
    return sched.build_inference_schedule(STEPS, device="cpu", dtype=torch.float32)


@pytest.fixture
def noise():
    gen = torch.Generator().manual_seed(0)
    return torch.randn(1, H, A, generator=gen)


def _serial(model, noise, timesteps, deltas):
    traj = [noise]
    x = noise
    for t, d in zip(timesteps, deltas):
        x = x + model.field(x, t.view(1)) * d
        traj.append(x)
    return torch.stack([s[0] for s in traj], dim=0)  # [S+1, H, A]


def _run(model, noise, schedule, **kwargs):
    timesteps, deltas = schedule
    kwargs.setdefault("window", STEPS)
    kwargs.setdefault("tol", 0.05)
    return model._infer_action_picard_loop(
        latents_action=noise, infer_timesteps_action=timesteps, infer_deltas_action=deltas,
        context=None, context_mask=None, video_kv_cache=None, attention_mask=None, video_seq_len=0, **kwargs)


# --------------------------------------------------------------------------- schedule

def test_schedule_endpoints_and_shift(schedule):
    timesteps, deltas = schedule
    assert timesteps.shape == deltas.shape == (STEPS,)
    assert float(timesteps[0]) == pytest.approx(1000.0)  # sigma_0 = 1
    assert float(deltas.sum()) == pytest.approx(-1.0, abs=1e-5)  # integrates sigma from 1 to 0
    assert bool((deltas < 0).all())
    u = 0.5  # shift * u / (1 + (shift - 1) * u) at u = 0.5
    t2, _ = WanContinuousFlowMatchScheduler(1000, SHIFT).build_inference_schedule(2, "cpu", torch.float32)
    assert float(t2[1]) == pytest.approx(1000 * SHIFT * u / (1 + (SHIFT - 1) * u), rel=1e-5)


# --------------------------------------------------------------------------- Eq. 3-5

def test_picard_with_zero_tolerance_equals_serial_euler(schedule, noise):
    model = _Sampler(_velocity)
    ref = _serial(model, noise, *schedule)
    out, stats = _run(model, noise, schedule, tol=0.0)
    torch.testing.assert_close(out[0], ref[-1], atol=1e-5, rtol=0)
    assert stats["init_mode"] == "linear"
    assert sum(r["stride"] for r in stats["rounds"]) == STEPS
    assert stats["final_trajectory"].shape == (STEPS + 1, H, A)


def test_picard_matches_serial_within_tolerance(schedule, noise):
    model = _Sampler(_velocity)
    ref = _serial(model, noise, *schedule)
    out, stats = _run(model, noise, schedule, tol=0.05)
    assert (out[0] - ref[-1]).abs().max() < 0.05
    assert stats["nfe"] <= STEPS + 1  # never worse than serial (+1 draft call)
    assert stats["picard_window"] == STEPS


def test_picard_collapses_to_one_batched_call_when_draft_is_exact(schedule, noise):
    model = _Sampler(_constant_velocity)
    ref = _serial(model, noise, *schedule)
    out, stats = _run(model, noise, schedule, tol=0.05)
    torch.testing.assert_close(out[0], ref[-1], atol=1e-5, rtol=0)
    assert stats["num_rounds"] == 1 and stats["mean_stride"] == STEPS
    assert stats["nfe"] == 2  # one draft call + one batched fixed-point sweep


def test_picard_window_smaller_than_schedule(schedule, noise):
    model = _Sampler(_velocity)
    ref = _serial(model, noise, *schedule)
    out, stats = _run(model, noise, schedule, window=4, tol=0.0)
    torch.testing.assert_close(out[0], ref[-1], atol=1e-5, rtol=0)
    assert all(r["p"] <= 4 for r in stats["rounds"])


# --------------------------------------------------------------------------- Eq. 6-7

def _warm(history, schedule, noise, replan):
    timesteps, _ = schedule
    return FastWAM._picard_noise_aligned_warm_trajectory(
        init_trajectory=history, infer_timesteps_action=timesteps, num_train_timesteps=1000,
        num_steps=STEPS, ref=noise, replan_steps=replan)[0]


def _history(seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(STEPS + 1, H, A, generator=gen)


@pytest.mark.parametrize("replan", [1, 10, 24, 31])
def test_noise_aligned_transfer_endpoints(schedule, noise, replan):
    hist = _history()
    warm = _warm(hist, schedule, noise, replan)
    assert warm.shape == hist.shape
    torch.testing.assert_close(warm[0], noise[0], atol=1e-6, rtol=0)  # sigma_0 = 1 -> fresh noise
    tail = H - replan
    idx = torch.arange(H)
    idx[:tail] += replan
    torch.testing.assert_close(warm[-1], hist[-1][idx], atol=1e-6, rtol=0)  # sigma_S = 0 -> shifted clean actions


def test_noise_aligned_transfer_keeps_deviation_from_linear_interpolation(schedule, noise):
    timesteps, _ = schedule
    sigma = torch.cat((timesteps, timesteps.new_zeros(1))) / 1000.0
    replan = 10
    hist = _history()
    warm = _warm(hist, schedule, noise, replan)
    idx = torch.arange(H)
    idx[:H - replan] += replan
    src = hist[:, idx]

    def deviation(traj):
        lin = sigma.view(-1, 1, 1) * traj[0] + (1 - sigma.view(-1, 1, 1)) * traj[-1]
        return traj - lin

    torch.testing.assert_close(deviation(warm), deviation(src), atol=1e-5, rtol=0)


def test_noise_aligned_transfer_is_a_pure_shift_when_noise_already_matches(schedule):
    # If eps_{n,h} == eps_{n-1,q(h)} the alignment term of Eq. 7 vanishes and only the shift remains.
    hist = _history()
    replan = H // 2
    idx = torch.arange(H)
    idx[:H - replan] += replan
    warm = _warm(hist, schedule, hist[0][idx].unsqueeze(0), replan=replan)
    torch.testing.assert_close(warm, hist[:, idx], atol=1e-6, rtol=0)


@pytest.mark.parametrize("replan", [0, H, H + 3])
def test_noise_aligned_transfer_rejects_invalid_replan(schedule, noise, replan):
    with pytest.raises(ValueError):
        _warm(_history(), schedule, noise, replan)


def test_noise_aligned_transfer_rejects_bad_shape(schedule, noise):
    with pytest.raises(ValueError):
        _warm(_history()[:-1], schedule, noise, 8)


def test_warm_start_requires_replan_steps(schedule, noise):
    model = _Sampler(_velocity)
    with pytest.raises(ValueError):
        _run(model, noise, schedule, init_trajectory=_history())


def test_warm_started_solve_matches_serial(schedule, noise):
    model = _Sampler(_velocity)
    ref = _serial(model, noise, *schedule)
    prev_noise = torch.randn(1, H, A, generator=torch.Generator().manual_seed(5))
    history = _serial(model, prev_noise, *schedule)
    out, stats = _run(model, noise, schedule, tol=0.0, init_trajectory=history, warm_start_replan_steps=10)
    torch.testing.assert_close(out[0], ref[-1], atol=1e-5, rtol=0)
    assert stats["init_mode"] == "noise_aligned_history"


# --------------------------------------------------------------------------- Eq. 4 prefix residual

def _corrupted_warm_start(model, noise, schedule, prefix):
    """Exact trajectory everywhere except a large perturbation on the unexecuted tail."""
    ref = _serial(model, noise, *schedule)
    warm = ref.clone()
    warm[1:, prefix:] += 1.0
    return ref, warm


def _run_with_fixed_warm(model, noise, schedule, warm, **kwargs):
    model._picard_noise_aligned_warm_trajectory = lambda **_: (warm, {})
    return _run(model, noise, schedule, init_trajectory=warm, warm_start_replan_steps=1, tol=0.05, **kwargs)


def test_prefix_residual_ignores_unexecuted_tail(schedule, noise):
    r = 10
    model = _Sampler(_velocity)
    ref, warm = _corrupted_warm_start(model, noise, schedule, r)
    out, stats = _run_with_fixed_warm(model, noise, schedule, warm, threshold_action_steps=r)
    assert stats["picard_threshold_action_steps"] == r
    assert stats["num_rounds"] == 1 and stats["nfe"] == 1  # prefix is already a fixed point: one batched call
    torch.testing.assert_close(out[0][:r], ref[-1][:r], atol=1e-5, rtol=0)


def test_full_residual_sees_tail_error_and_keeps_iterating(schedule, noise):
    r = 10
    model = _Sampler(_velocity)
    ref, warm = _corrupted_warm_start(model, noise, schedule, r)
    out, stats = _run_with_fixed_warm(model, noise, schedule, warm, threshold_action_steps=None)
    assert stats["picard_threshold_action_steps"] == H
    assert stats["num_rounds"] > 1 and stats["nfe"] > 1
    full_tail_err = (out[0][r:] - ref[-1][r:]).abs().max()
    assert full_tail_err < 0.1  # converges everywhere (up to the tau-level tolerance) ...
    prefix_out, _ = _run_with_fixed_warm(_Sampler(_velocity), noise, schedule, warm, threshold_action_steps=r)
    prefix_tail_err = (prefix_out[0][r:] - ref[-1][r:]).abs().max()
    assert prefix_tail_err > 5 * full_tail_err  # ... whereas the prefix criterion leaves the tail unrefined


@pytest.mark.parametrize("bad", [0, -1, H + 1])
def test_threshold_steps_out_of_range(schedule, noise, bad):
    with pytest.raises(ValueError):
        _run(_Sampler(_velocity), noise, schedule, threshold_action_steps=bad)
