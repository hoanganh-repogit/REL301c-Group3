"""
model.py
Q-Network kien truc MLP don gian (Linear_QNet) va QTrainer thuc hien buoc cap nhat
Q-value theo phuong trinh Bellman:

    Q(s, a) <- r + gamma * max_a' Q_target(s', a')      (neu khong terminal)
    Q(s, a) <- r                                          (neu terminal)

Cai tien so voi ban goc:
  - Target Network rieng (target_model) de tinh bootstrap target, tranh
    hien tuong "moving target" khien DQN training khong on dinh
  - Huber loss (SmoothL1) thay vi MSE -> it nhay voi outlier Q-value
  - Ho tro device (CPU/GPU)
  - Shape assertions o cac diem chuyen doi tensor quan trong
  - Gradient clipping tranh exploding gradients
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Linear_QNet(nn.Module):
    """MLP 2 lop anh xa state (input_size,) -> Q-values (output_size,)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_size) hoac (input_size,)
        assert x.shape[-1] == self.input_size, (
            f"Linear_QNet.forward: expected last dim {self.input_size}, got {x.shape[-1]}"
        )
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        assert x.shape[-1] == self.output_size
        return x

    def save(self, file_name: str = "model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """Thuc hien 1 buoc cap nhat gradient theo TD-error cua Q-learning."""

    def __init__(
        self,
        model: Linear_QNet,
        lr: float,
        gamma: float,
        target_update_freq: int = 100,
        device: str = None,
    ):
        self.lr = lr
        self.gamma = gamma
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = model.to(self.device)

        # Target network: kien truc giong het model chinh, khoi tao cung trong so,
        # nhung KHONG duoc cap nhat truc tiep bang backprop - chi sync dinh ky.
        self.target_model = Linear_QNet(
            model.input_size, model.hidden_size, model.output_size
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.target_update_freq = target_update_freq
        self.train_step_count = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

    def update_target_network(self):
        """Dong bo trong so: target_model = model (hard update)."""
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        # Shape hien tai: (x,) neu single sample, (n, x) neu batch

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        batch_size = state.shape[0]
        assert state.shape == (batch_size, self.model.input_size)
        assert next_state.shape == (batch_size, self.model.input_size)
        assert action.shape == (batch_size, self.model.output_size)
        assert reward.shape == (batch_size,)
        assert len(done) == batch_size

        # 1) Q(s, a) hien tai theo model dang duoc train
        pred = self.model(state)  # (batch_size, output_size)

        target = pred.clone()
        with torch.no_grad():
            # 2) Q_target(s', a') lay tu target network, khong lan truyen gradient
            next_q_values = self.target_model(next_state)  # (batch_size, output_size)
            max_next_q = torch.max(next_q_values, dim=1).values  # (batch_size,)

        for idx in range(batch_size):
            # Bellman equation: Q_new = r + gamma * max_a' Q_target(s', a')
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * max_next_q[idx]
            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()