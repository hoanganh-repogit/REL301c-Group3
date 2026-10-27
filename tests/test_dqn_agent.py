"""
Shape and basic logic checks for DuelingQNetwork + DoubleDQNAgent. These tests
do NOT run full training (scripts/train_final_dqn.py does that); they only check
shapes and dtypes and that nothing crashes.
"""

import numpy as np
import torch

from src.agents.dqn.agent import DoubleDQNAgent, DQNConfig
from src.agents.dqn.network import DuelingQNetwork
from src.agents.dqn.replay_buffer import ReplayBuffer
from src.envs.snake_env import ACTION_SPACE
from src.envs.state_encoder import STATE_DIM


def test_replay_buffer_uses_configured_capacity():
    config = DQNConfig(batch_size=8, buffer_size=123, min_buffer_size_before_train=8)
    agent = DoubleDQNAgent(config)
    assert agent.buffer.capacity == 123


def test_network_output_shape():
    net = DuelingQNetwork()
    batch = torch.randn(8, STATE_DIM)
    q_values = net(batch)
    assert q_values.shape == (8, ACTION_SPACE)


def test_network_single_sample():
    net = DuelingQNetwork()
    single = torch.randn(1, STATE_DIM)
    q_values = net(single)
    assert q_values.shape == (1, ACTION_SPACE)


def test_replay_buffer_push_and_sample():
    buffer = ReplayBuffer(capacity=100)
    for i in range(50):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        buffer.push(state, action=i % ACTION_SPACE, reward=1.0, next_state=next_state, done=False)

    assert len(buffer) == 50
    states, actions, rewards, next_states, dones = buffer.sample(16, device=torch.device("cpu"))
    assert states.shape == (16, STATE_DIM)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert next_states.shape == (16, STATE_DIM)
    assert dones.shape == (16,)


def test_replay_buffer_clear():
    buffer = ReplayBuffer(capacity=100)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    buffer.push(state, 0, 1.0, state, False)
    assert len(buffer) == 1
    buffer.clear()
    assert len(buffer) == 0


def test_replay_buffer_state_round_trip():
    source = ReplayBuffer(capacity=100)
    for index in range(12):
        state = np.full(STATE_DIM, index, dtype=np.float32)
        source.push(
            state,
            index % ACTION_SPACE,
            float(index),
            state + 1,
            index % 2 == 0,
            level="level4",
        )

    restored = ReplayBuffer(capacity=100)
    restored.load_state_dict(source.state_dict())
    assert len(restored) == len(source)
    assert restored.buffer[-1].level == "level4"
    np.testing.assert_array_equal(restored.buffer[-1].state, source.buffer[-1].state)


def test_agent_act_returns_valid_action():
    agent = DoubleDQNAgent(DQNConfig())
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action = agent.act(state)
    assert isinstance(action, int)
    assert 0 <= action < ACTION_SPACE


def test_agent_act_greedy_deterministic():
    """With greedy=True, the same state must always give the same action (no randomness)."""
    agent = DoubleDQNAgent(DQNConfig())
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action1 = agent.act(state, greedy=True)
    action2 = agent.act(state, greedy=True)
    assert action1 == action2


def test_agent_update_returns_none_when_buffer_too_small():
    config = DQNConfig(min_buffer_size_before_train=500)
    agent = DoubleDQNAgent(config)
    state = np.random.randn(STATE_DIM).astype(np.float32)
    agent.store(state, 0, 1.0, state, False)  # only one transition, not enough
    assert agent.update() is None


def test_agent_update_runs_when_buffer_enough():
    config = DQNConfig(min_buffer_size_before_train=10, batch_size=8)
    agent = DoubleDQNAgent(config)
    for _ in range(20):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        agent.store(state, 0, 1.0, next_state, False)

    loss = agent.update()
    assert loss is not None
    assert loss >= 0.0  # the loss is never negative


def test_epsilon_schedule_decays_to_end():
    config = DQNConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100)
    agent = DoubleDQNAgent(config)
    assert agent.epsilon() == 1.0  # no steps yet -> epsilon_start

    state = np.random.randn(STATE_DIM).astype(np.float32)
    for _ in range(100):
        agent.act(state)  # each non-greedy act() increments total_env_steps by 1

    assert abs(agent.epsilon() - 0.1) < 1e-6  # fully decayed -> epsilon_end


