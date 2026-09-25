"""
Kiểm tra shape và logic cơ bản của DuelingQNetwork + DoubleDQNAgent —
KHÔNG chạy full training (đã có sanity_train_dqn_level1.py cho việc đó),
chỉ đảm bảo mọi thứ chạy đúng shape/kiểu dữ liệu, không crash.
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


def test_agent_act_returns_valid_action():
    agent = DoubleDQNAgent(DQNConfig())
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action = agent.act(state)
    assert isinstance(action, int)
    assert 0 <= action < ACTION_SPACE


def test_agent_act_greedy_deterministic():
    """Với greedy=True, cùng 1 state phải luôn ra cùng 1 action (không random)."""
    agent = DoubleDQNAgent(DQNConfig())
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action1 = agent.act(state, greedy=True)
    action2 = agent.act(state, greedy=True)
    assert action1 == action2


def test_agent_update_returns_none_when_buffer_too_small():
    config = DQNConfig(min_buffer_size_before_train=500)
    agent = DoubleDQNAgent(config)
    state = np.random.randn(STATE_DIM).astype(np.float32)
    agent.store(state, 0, 1.0, state, False)  # chỉ 1 transition, chưa đủ
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
    assert loss >= 0.0  # MSE loss luôn không âm


def test_epsilon_schedule_decays_and_resets():
    config = DQNConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100)
    agent = DoubleDQNAgent(config)
    assert agent.epsilon() == 1.0  # chưa bước nào -> epsilon_start

    state = np.random.randn(STATE_DIM).astype(np.float32)
    for _ in range(100):
        agent.act(state)  # mỗi lần act (không greedy) tăng total_env_steps 1 đơn vị

    assert abs(agent.epsilon() - 0.1) < 1e-6  # đã decay hết -> epsilon_end

    agent.reset_epsilon_schedule()
    assert agent.epsilon() == 1.0  # reset về start -- dùng khi lên level mới (Giai đoạn 3)
