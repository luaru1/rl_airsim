import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size = 64, buffer_size=100000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 디바이스 설정
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)        # Q-network 선언
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)   # 목표값 계산을 위한 이전 상태의 Q-network 선언
        self.target_net.load_state_dict(self.q_net.state_dict())    # Q-network의 가중치를 target network에 복사
        self.target_net.eval()              # target network를 추론 모드로 전환(드롭아웃, 배치정규화 등을 비활성)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr) # Optimizer 선택: Adam
        self.loss_fn = nn.MSELoss()         # Loss function 선택: Mean Squared Error

        self.replay_buffer = deque(maxlen=buffer_size)  # 이전까지의 경험을 저장. 일정 수 이상 모이면 배치 샘플링해 학습
        self.batch_size = batch_size
        self.gamma = gamma                  # 미래의 최대 예상 보상을 얼마나 반영할 지
        self.action_dim = action_dim

        self.epsilon = 1.0                  # 초기 탐험 확률 (처음에는 100% 랜덤으로 행동)
        self.epsilon_decay = 0.995          # 탐험 확률의 감소 비율 (일반적으로 에피소드마다 감소)
        self.epsilon_min = 0.05             # 탐험 확률의 최저 한계
    
    def select_action(self, state):
        # 확률적으로 탐험을 진행(무작위로 행동 선택)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        # 그 외에는 현재 상태에 따라 가장 Q값이 높을 것으로 예상되는 행동을 선택
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)   # unsqueeze로 차원을 추가해 state에 저장
        with torch.no_grad():               # Q값을 예측하는 동안 gradiant 계산을 비활성화
            q_values = self.q_net(state)    # Q값 예측
        return q_values.argmax().item()     # 가장 큰 Q값을 가지는 행동의 index를 반환

    def store_transition(self, state, action, reward, next_state, done):
        # 리플레이 버퍼에 경험을 저장
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        # 리플레이 버퍼에 충분한 경험이 쌓이지 않았다면 return
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 리플레이 버퍼에서 batch_size만큼 샘플링해 state, action, reward, next_state, done을 구분
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 각각을 텐서로 변환해 디바이스에 전달
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)

        # 현재 Q값 계산
        q_values = self.q_net(states).gather(1, actions)
        # target network로 다음 상태에서의 Q값 계산
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones.float())   # Target Q값 계산. 종료된 경우 0이 되어야하므로 (1 - dones)를 곱해줌
        
        # 손실 계산
        loss = self.loss_fn(q_values, target_q)

        # 손실을 바탕으로 현재 Q값과 타겟 Q값의 차이를 줄이도록 학습
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 탐험 확률 감소
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def update_target_network(self):
        # 일정 간격마다 현재 Q-network의 파라미터를 target-network에 복사
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path):
        # 현재까지 학습된 Q-network, target-network, optimizer, epsilon을 저장
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        # 저장된 모델을 로드
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']