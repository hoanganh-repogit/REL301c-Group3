"""
Kiểm tra shape và logic cơ bản của ActorCriticNetwork + PPOAgent + RolloutBuffer.
Không chạy full training (đã có sanity_train_ppo_level1.py).
"""

import numpy as np
import torch

from src.agents.ppo.agent import PPOAgent, PPOConfig
from src.agents.ppo.buffer import RolloutBuffer
from src.agents.ppo.network import ActorCriticNetwork
from src.envs.snake_env import ACTION_SPACE
from src.envs.state_encoder import STATE_DIM


def test_network_forward_shape():
    net = ActorCriticNetwork()
    batch = torch.randn(8, STATE_DIM)
    logits, value = net(batch)
    assert logits.shape == (8, ACTION_SPACE)
    assert value.shape == (8,)


def test_network_act_shape():
    net = ActorCriticNetwork()
    single = torch.randn(1, STATE_DIM)
    action, log_prob, value = net.act(single)
    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert value.shape == (1,)
    assert 0 <= int(action.item()) < ACTION_SPACE


def test_network_evaluate_actions_shape():
    net = ActorCriticNetwork()
    states = torch.randn(16, STATE_DIM)
    actions = torch.randint(0, ACTION_SPACE, (16,))
    log_probs, entropy, values = net.evaluate_actions(states, actions)
    assert log_probs.shape == (16,)
    assert entropy.shape == (16,)
    assert values.shape == (16,)
    assert (entropy >= 0).all()  # entropy của phân phối rời rạc luôn không âm


def test_rollout_buffer_gae_all_done_equals_monte_carlo_return():
    """Case biên: nếu episode nào cũng done=True ngay (T=1 mỗi lần), GAE phải suy biến
    về đúng công thức return 1 bước: A_t = r_t - V(s_t), return_t = r_t."""
    buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    buffer.add(state, action=0, reward=5.0, done=True, log_prob=-0.5, value=2.0)

    advantages, returns = buffer.compute_returns_and_advantages(last_value=0.0)
    assert advantages.shape == (1,)
    assert returns.shape == (1,)
    # done=True -> not_done=0 -> delta = r_t - V(s_t) = 5.0 - 2.0 = 3.0, gae = delta (vì not_done=0 chặn đệ quy)
    assert abs(advantages[0] - 3.0) < 1e-5
    assert abs(returns[0] - 5.0) < 1e-5  # return = advantage + value = 3.0 + 2.0 = 5.0


def test_rollout_buffer_reset_clears_data():
    buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    buffer.add(state, 0, 1.0, False, -0.1, 0.5)
    assert len(buffer) == 1
    buffer.reset()
    assert len(buffer) == 0


def test_agent_act_returns_valid_action():
    agent = PPOAgent(PPOConfig())
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action, log_prob, value = agent.act(state)
    assert isinstance(action, int)
    assert 0 <= action < ACTION_SPACE
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_agent_act_greedy_deterministic():
    agent = PPOAgent(PPOConfig())
    state = np.random.randn(STATE_DIM).astype(np.float32)
    a1 = agent.act_greedy(state)
    a2 = agent.act_greedy(state)
    assert a1 == a2


def test_agent_update_runs_and_resets_buffer():
    config = PPOConfig(rollout_steps=32, minibatch_size=8, k_epochs=2)
    agent = PPOAgent(config)

    for _ in range(32):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action, log_prob, value = agent.act(state)
        done = bool(np.random.rand() < 0.1)
        agent.store(state, action, reward=1.0, done=done, log_prob=log_prob, value=value)

    assert len(agent.buffer) == 32
    stats = agent.update(last_value=0.0)

    assert "policy_loss" in stats and "value_loss" in stats and "entropy" in stats
    assert len(agent.buffer) == 0  # PPO PHẢI reset buffer sau update (on-policy)


def test_bootstrap_value_zero_when_done():
    agent = PPOAgent(PPOConfig())
    state = np.random.randn(STATE_DIM).astype(np.float32)
    assert agent.bootstrap_value(state, done=True) == 0.0
