import sys, os
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# 경로 설정 및 디렉토리 생성
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'config.yaml')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dqn_model.pt')
LOG_DIR = os.path.join(BASE_DIR, 'logs', 'train')

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

from agents.dqn_agent import DQNAgent
from envs.airsim_env import AirSimCarEnv

# 텐서보드 로그 기록기 생성
writer = SummaryWriter(LOG_DIR)

# 설정 파일 로드
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# 시드 고정
seed = config['misc']['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 환경 초기화
max_step_per_episode = config['train']['max_steps_per_episode']
env_config = {
    **config['env'],
    'max_step': max_step_per_episode
}
env = AirSimCarEnv(env_config)

# 에이전트 초기화
state_dim = env._get_observation().shape[0]
action_dim = len(env.action_space)
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=config['train']['gamma'],
    lr=config['train']['lr'],
    batch_size=config['train']['batch_size'],
    buffer_size=config['train']['buffer_size']
)

print(f'현재 디바이스: {agent.device}')

# 에피소드 루프 조건 설정
num_episodes = config['train']['num_episodes']
target_update_freq = config['train']['target_update_freq']
save_interval = config['train']['save_interval']

# 에피소드 루프 진행
for episode in range(1, num_episodes + 1):
    # 환경 및 보상 초기화
    state = env.reset()
    total_reward = 0

    # 에피소드 내 스텝 진행
    for t in range(max_step_per_episode):
        # 행동 선택
        action = agent.select_action(state)
        # 시뮬레이터 실행
        next_state, reward, done, info = env.step(action)
        # 행동 결과 저장 및 학습
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        # 상태 및 보상 업데이트
        state = next_state
        total_reward += reward
        # 조기 종료 조건 충족 시 한 에피소드 종료
        if done:
            break
    
    # 정해진 주기마다 target Q-network 갱신
    if episode % target_update_freq == 0:
        agent.update_target_network()
    
    # 정해진 주기마다 모델 저장
    if episode % save_interval == 0:
        agent.save_model(MODEL_PATH)
    
    # 실시간 진행 상황 출력
    if config['misc']['verbose']:
        print(f"Episode {episode} | Reward: {round(total_reward, 2)} | Done: {info['done_reason']}")
    
    # 텐서보드 로그 작성
    writer.add_scalar("Reward/Total", total_reward, episode)
    writer.add_scalar("Episode/StepCount", info['step'], episode)

    reward_detail = info['reward_detail']
    for k, v in reward_detail.items():
        writer.add_scalar(f"RewardDetail/{k}", v, episode)

    for reason in info['done_reason']:
        writer.add_scalar(f"DoneReason/{reason}", 1, episode)

print("Training complete.")

# 텐서보드 로그 기록기 종료
writer.close()