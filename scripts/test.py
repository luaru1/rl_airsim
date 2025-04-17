import sys, os
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# 경로 설정 및 디렉토리 생성
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'config.yaml')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dqn_model.pt')
LOG_DIR = os.path.join(BASE_DIR, 'logs', 'test')

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

# 모델 로드
agent.load_model(MODEL_PATH)
# 탐험 확률 0으로 변경: 항상 환경에 따라 action을 선택하도록 설정
agent.epsilon = 0.0

# 에피소드 루프 조건 설정 및 루프 진행
num_episodes = config['test']['num_episodes']
for episode in range(1, num_episodes + 1):
    # 환경 및 보상 초기화
    state = env.reset()
    total_reward = 0

    # 에피소드 내 스텝 진행
    for t in range(max_step_per_episode):
        # 행동 선택 및 결과 반환
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        # 상태 및 보상 업데이트
        state = next_state
        total_reward += reward

        # 조기 종료 조건 충족 시 한 에피소드 종료
        if done:
            break
    
    # 실시간 진행 상황 출력
    print(f'[TEST] Episode {episode} | Reward: {round(total_reward, 2)} | Steps: {info["step"]} | Done: {info["done_reason"]}')

    # 텐서보드 로그 작성
    writer.add_scalar('Test/RewardTotal', total_reward, episode)
    writer.add_scalar('Test/StepCount', info['step'], episode)

    reward_detail = info['reward_detail']
    for k, v in reward_detail.items():
        writer.add_scalar(f'Test/RewardDetail/{k}', v, episode)

    for reason in info['done_reason']:
        writer.add_scalar(f'Test/DoneReason/{reason}', 1, episode)

print('Testing complete.')

# 텐서보드 로그 기록기 종료
writer.close()