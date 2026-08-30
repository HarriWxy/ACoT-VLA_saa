# ruff: noqa: SLF001

import numpy as np

from openpi.serving import websocket_policy_server_sample as _server


class _Policy:
    def infer(self, obs, **kwargs):
        del obs, kwargs
        return {"actions": np.zeros((1, 2), dtype=np.float32)}


class _Dataset:
    def __init__(self):
        self.frames: list[dict] = []

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)


def _make_server() -> _server.WebsocketPolicyServer:
    policy = _Policy()
    return _server.WebsocketPolicyServer(
        policy=policy,
        config=_server.ServerConfig(policy=policy, enable_data_collection=True),
    )


def test_parse_request_separates_transition_feedback_from_policy_observation():
    request = _server._parse_request(
        {
            "obs": {"state": np.array([1.0, 2.0], dtype=np.float32)},
            "task": "walk forward",
            "reward": np.array([3.0], dtype=np.float32),
            "executed_action": np.array([0.1, 0.2], dtype=np.float32),
            "terminated": np.array([True]),
        }
    )

    assert request.task == "walk forward"
    assert request.done
    assert set(request.observation) == {"state", "prompt"}
    np.testing.assert_allclose(request.reward, [3.0])
    np.testing.assert_allclose(request.executed_action, [0.1, 0.2])


def test_detect_obs_features_supports_chw_images_and_configured_state_layout():
    features, state_keys, image_keys = _server._detect_obs_features(
        {
            "state": np.zeros(2, dtype=np.float32),
            "proprio_dyn": np.ones(3, dtype=np.float32),
            "image_base": np.zeros((3, 32, 40), dtype=np.uint8),
        },
        configured_state_keys=("state", "proprio_dyn"),
        configured_image_keys=None,
    )

    assert state_keys == ["state", "proprio_dyn"]
    assert image_keys == ["image_base"]
    assert features["observation.state"]["shape"] == (5,)
    assert features["observation.images.image_base"]["shape"] == (32, 40, 3)


def test_output_noise_is_written_to_the_returned_action_chunk():
    server = _make_server()
    server._apply_output_noise = lambda action: action + 10.0  # type: ignore[method-assign]

    outgoing, first_action = server._prepare_action_result(
        {"actions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
        apply_output_noise=True,
    )

    np.testing.assert_allclose(first_action, [11.0, 12.0])
    np.testing.assert_allclose(outgoing["actions"], [[11.0, 12.0], [3.0, 4.0]])


def test_pending_frame_uses_next_request_reward_and_executed_action():
    server = _make_server()
    dataset = _Dataset()
    server._dataset = dataset
    server._initialized = True
    server._action_dim = 2
    server._state_keys = ["state"]
    server._image_keys = []
    server._features = {"observation.state": {"shape": (2,)}}

    pending = _server._PendingFrame(
        observation={"state": np.array([1.0, 2.0], dtype=np.float32)},
        action=np.array([0.1, 0.2], dtype=np.float32),
        task="walk",
    )
    request = _server._parse_request(
        {
            "state": np.array([3.0, 4.0], dtype=np.float32),
            "reward": np.array([5.0], dtype=np.float32),
            "executed_action": np.array([0.3, 0.4], dtype=np.float32),
        }
    )

    server._commit_pending_frame(pending, request)

    assert len(dataset.frames) == 1
    frame = dataset.frames[0]
    np.testing.assert_allclose(frame["observation.state"], [1.0, 2.0])
    np.testing.assert_allclose(frame["action"], [0.3, 0.4])
    np.testing.assert_allclose(frame["reward"], [[5.0]])
