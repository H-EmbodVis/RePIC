import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import Pi0
from openpi.models.pi0 import _picard_noise_aligned_warm_trajectory
from openpi.models.pi0 import _picard_prefix_integrate
from openpi.models.pi0 import _picard_stride


def test_picard_prefix_and_stride():
    sample = jnp.array([[[1.0]]])
    velocities = jnp.array([[[[2.0]]], [[[3.0]]]])
    states = _picard_prefix_integrate(sample, velocities, jnp.array([-0.5, -0.5]))
    np.testing.assert_allclose(states.reshape(-1), [1.0, 0.0, -1.5])
    stride, error, threshold, _ = _picard_stride(
        states.at[1].set(0.2), states, jnp.array([1.0, 0.5, 0.0]), 2, 0.1
    )
    assert int(stride) == 1
    assert float(error[1]) > float(threshold[1])


class _FakePi05:
    action_horizon = 2
    action_dim = 1

    def _predict_velocity_with_cache(self, observation, x, time, prefix_mask, kv_cache):
        del observation, prefix_mask
        assert kv_cache[0].shape[1] == x.shape[0]
        return x * 0.2 + time[:, None, None]


def test_picard_trace_reaches_all_steps():
    model = _FakePi05()
    result = jax.jit(
        lambda initial_noise: Pi0._sample_actions_picard(
            model, {"state": jnp.zeros((1, 1))}, initial_noise,
            jnp.ones((1, 1), dtype=jnp.bool_),
            (jnp.zeros((1, 1, 1, 1, 1)), jnp.zeros((1, 1, 1, 1, 1))),
            num_steps=5, window=3, tol=0.1, warm_start_trajectory=None,
            warm_start_replan_steps=None, threshold_action_steps=None,
            return_sampling_stats=True, return_sampling_trace=True,
        )
    )(jnp.ones((1, 2, 1)))
    stats = result["sampling_stats"]
    rounds = int(stats["num_rounds"])
    assert int(stats["nfe"]) == 1 + rounds
    assert int(np.asarray(stats["round_stride"][:rounds]).sum()) == 5
    np.testing.assert_allclose(result["sampling_trace"]["final_trajectory"][-1], result["actions"])


def test_noise_aligned_warm_start_shifts_tail():
    trajectory = jnp.arange(3 * 1 * 4, dtype=jnp.float32).reshape(3, 1, 4, 1)
    warm = _picard_noise_aligned_warm_trajectory(
        trajectory, jnp.array([1.0, 0.5]), jnp.full((1, 4, 1), -7.0), 2
    )
    np.testing.assert_allclose(warm[0], -7.0)
    np.testing.assert_allclose(warm[-1, :, :2], trajectory[-1, :, 2:])
