import airsim
import numpy as np
import time
import math

from utils.angle_utils import quaternion_to_euler

class AirSimCarEnv:
    def __init__(self):
        # AirSim 클라이언트 연결
        self.client = airsim.CarClient()        # 시뮬레이터와 코드 사이의 연결 객체
        self.client.confirmConnection()         # 통신 가능성 체크
        self.client.enableApiControl(True)      # API 제어 권한 활성화
        self.client.armDisarm(True)             # 제어 명령 수신 활성화

        # 차량 제어 객체
        self.car_controls = airsim.CarControls()

        # 차량 초기 위치 저장
        self.initial_pose = self.client.simGetVehiclePose()

        # Action space (차량이 할 수 있는 행동) 정의
        self.throttle_vals = [0.3, 0.5, 0.7]    # 엑셀 페달을 얼마나 세게 밟을지
        self.steering_vals = [-0.5, 0.0, 0.5]   # 핸들의 방향을 얼마나 돌릴지
        self.gear_vals = [1, -1]                # 전진 기어인지 후진 기어인지
        self.action_space = [
            (t, s, g)
            for t in self.throttle_vals
            for s in self.steering_vals
            for g in self.gear_vals
        ]

        # 시뮬레이션 Step 정보
        self.step_counter = 0
        self.max_step = 5000

        # 직전 시점에서의 위치
        self.prev_position = None

        # 차량의 한계 위치
        self.max_x_val = 5000
        self.max_y_val = 5000

        # 차량의 초저속 주행이 얼마나 지속되는지 확인
        self.slow_state_window = []     # 최근 N step동안 초저속 주행을 했는지 여부 저장
        self.window_n = 50              # 초저속 주행 여부 배열 크기
        self.low_speed_limit = 1.5      # 초저속 주행 속도 한계
    
    def reset(self):
        # 차량 초기화
        self.client.reset()                     # 시뮬레이터 환경 초기화
        self.client.enableApiControl(True)      # reset에 의해 초기화된 API 제어 권한 활성화
        self.client.armDisarm(True)             # reset에 의해 초기화된 제어 명령 수신 활성화
        self.client.simSetVehiclePose(self.initial_pose, True)  # 차량을 초기 위치로 이동

        # 차량 정지 상태 설정
        self.car_controls.throttle = 0.0        # 엑셀 페달 초기화
        self.car_controls.steering = 0.0        # 핸들 방향 초기화
        self.car_controls.is_manual_gear = True
        self.car_controls.manual_gear = 1       # 전진 기어로 초기화
        self.car_controls.brake = 1.0           # 브레이크를 최대로 밟음
        self.client.setCarControls(self.car_controls)   # 차량에 상태를 적용

        # Step 초기화
        self.step_counter = 0

        # 직전 시점에서의 위치 초기화
        self.prev_position = self.client.getCarState().kinematics_estimated.position

        # 초저속 주행 여부 배열 초기화
        self.slow_state_window = []

        time.sleep(0.5)                         # 적용 대기

        return self._get_observation()          # 현재 차량 상태 반환

    def step(self, action_idx):
        # 1개 Step 진행
        self.step_counter += 1

        # 행동 적용
        throttle, steering, gear = self.action_space[action_idx]    # Action space에서 행동 선택
        self.car_controls.throttle = throttle
        self.car_controls.steering = steering
        self.car_controls.is_manual_gear = True
        self.car_controls.manual_gear = gear
        self.car_controls.brake = 0.0
        self.client.setCarControls(self.car_controls)

        time.sleep(0.1)                         # 적용 대기

        obs = self._get_observation()           # 현재 차량 상태 반환
        
        reward_detail = self._compute_reward()         # 주행 성과에 따른 보상 계산
        
        if 'collision_penalty' in reward_detail:
            total_reward = reward_detail['collision_penalty']
        else:
            total_reward = sum(reward_detail.values())

        done_info = self._check_done()          # 종료 조건 계산
        done = done_info['done']
        info = {                                # 디버깅용 보조 정보. 필요에 따라 추가. 학습에 이용되지 않음
            'step': self.step_counter,
            'reward_detail': reward_detail,
            'done_reason': [key for key in ['collision', 'out_of_bounds', 'too_slow', 'timeout'] if done_info[key]]
        }                               

        return obs, total_reward, done, info
    
    def _get_observation(self):
        # 현재 차량 상태 확인
        state = self.client.getCarState()       # 현재 차량 상태 반환
        pos = state.kinematics_estimated.position   # 차량의 3차원 위치를 반환
        vel = state.speed                       # 차량의 현재 속도를 실수형으로 반환 (단위: m/s)
        orientation = state.kinematics_estimated.orientation    # 차량의 방향 정보
        roll, pitch, yaw = quaternion_to_euler(orientation)  # 차량의 방향을 오일러 각도(rad)로 반환

        # 관측값을 numpy 배열로 구성해 반환. DQN의 state 벡터
        return np.array([pos.x_val, pos.y_val, pos.z_val, vel, roll, pitch, yaw], dtype=np.float32)
    
    def _compute_reward(self):
        state = self.client.getCarState()       # 현재 차량 상태 반환
        pos = state.kinematics_estimated.position   # 현재 차량의 위치 반환
        speed = state.speed                     # 현재 차량의 속도 반환
        collision = self.client.simGetCollisionInfo().has_collided  # 충돌 여부 확인

        # 충돌 페널티
        if collision:
            return {'collision_penalty': -100.0}    # 충돌시 큰 페널티 부여
        
        # 속도 보상
        target_speed = 10.0                     # 목표 속도 10m/s == 36km/h
        speed_reward = -abs(speed - target_speed) + target_speed    # 목표 속도에 가까울수록 더 큰 보상 부여. target_speed가 최대 보상

        # 이동 거리 보상
        if self.prev_position is not None:
            dx = pos.x_val - self.prev_position.x_val
            dy = pos.y_val - self.prev_position.y_val
            dz = pos.z_val - self.prev_position.z_val
            step_distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        else:
            step_distance = 0.0
        distance_reward = step_distance * 10.0

        self.prev_position = pos                # 다음 계산을 위해 현재 위치 저장

        # 급회전 페널티
        steering_penalty = -abs(self.car_controls.steering) * 0.3    # 핸들 회전이 클수록 페널티 부여

        # 도로 이탈 페널티(미구현)
        boundary_penalty = 0.0

        return {
            'speed_reward': speed_reward,
            'distance_reward': distance_reward,
            'steering_penalty': steering_penalty,
            'boundary_penalty': boundary_penalty
        }
    
    def _check_done(self):
        # 충돌 시 종료
        collision = self.client.simGetCollisionInfo().has_collided

        # 차량 위치가 한계 범위를 벗어나면 종료
        pos = self.client.getCarState().kinematics_estimated.position
        out_of_bounds = abs(pos.x_val) > self.max_x_val or abs(pos.y_val) > self.max_y_val

        # 차량이 초저속으로 오래 지속되면 종료
        speed = self.client.getCarState().speed
        self.slow_state_window.append(speed < self.low_speed_limit)
        if len(self.slow_state_window) > self.window_n:
            self.slow_state_window.pop(0)
        
        if sum(self.slow_state_window) / len(self.slow_state_window) >= 0.9:
            too_slow = True
        else:
            too_slow = False

        # 최대 step에 도달하면 종료
        timeout = self.step_counter >= self.max_step

        return {
            'done': collision or out_of_bounds or too_slow or timeout,
            'collision': collision,
            'out_of_bounds': out_of_bounds,
            'too_slow': too_slow,
            'timeout': timeout
        }