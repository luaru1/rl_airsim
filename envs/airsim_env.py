import airsim
import numpy as np
import time
import math
from PIL import Image

from utils.angle_utils import quaternion_to_euler
from perception.lidar_processor import LidarProcessor
from perception.road_detector import RoadDetector
from planners.obstacle_avoider import ObstacleAvoider

class AirSimCarEnv:
    def __init__(self, env_config):
        # AirSim 클라이언트 연결
        self.client = airsim.CarClient()        # 시뮬레이터와 코드 사이의 연결 객체
        self.client.confirmConnection()         # 통신 가능성 체크
        self.client.enableApiControl(True)      # API 제어 권한 활성화
        self.client.armDisarm(True)             # 제어 명령 수신 활성화

        # 차량 제어 객체
        self.car_controls = airsim.CarControls()

        # 차량 초기 위치 저장
        self.initial_pose = self.client.simGetVehiclePose()
        self.init_x = self.initial_pose.position.x_val
        self.init_y = self.initial_pose.position.y_val
        self.init_z = self.initial_pose.position.z_val

        # Action space (차량이 할 수 있는 행동) 정의
        self.throttle_vals = [0.3, 0.5, 0.7]    # 엑셀 페달을 얼마나 세게 밟을지
        self.steering_vals = [-0.8, -0.5, 0.0, 0.5, 0.8]   # 핸들의 방향을 얼마나 돌릴지
        self.gear_vals = [1, -1]                # 전진 기어인지 후진 기어인지
        self.action_space = [
            (t, s, g)
            for t in self.throttle_vals
            for s in self.steering_vals
            for g in self.gear_vals
        ]

        # 시뮬레이션 Step 정보
        self.step_counter = 0
        self.max_step = env_config.get('max_step', 5000)

        # 차량 충돌 여부
        self.collision_detected = False

        # 최대 이동 거리
        self.max_dist = 0

        # 차량의 한계 위치
        self.max_x_val = env_config['max_x']
        self.max_y_val = env_config['max_y']

        # 차량의 목표 속도
        self.target_speed = env_config['reward_target_speed']

        # 차량의 초저속 주행이 얼마나 지속되는지 확인
        self.slow_state_window = []     # 최근 N step동안 초저속 주행을 했는지 여부 저장
        self.window_n = env_config['slow_window_size']  # 초저속 주행 여부 배열 크기
        self.low_speed_limit = env_config['low_speed_limit']    # 초저속 주행 속도 한계

        self.lidar_processor = LidarProcessor()
        self.road_detector = RoadDetector()
        self.avoider = ObstacleAvoider()
    
    def reset(self):
        # 차량 초기화
        self.client.reset()                     # 시뮬레이터 환경 초기화
        self.client.enableApiControl(True)      # reset에 의해 초기화된 API 제어 권한 활성화
        self.client.armDisarm(True)             # reset에 의해 초기화된 제어 명령 수신 활성화
        self.client.simSetVehiclePose(self.initial_pose, ignore_collision=True)  # 차량을 초기 위치로 이동

        # 차량 정지 상태 설정
        self.car_controls.throttle = 0.0        # 엑셀 페달 초기화
        self.car_controls.steering = 0.0        # 핸들 방향 초기화
        self.car_controls.is_manual_gear = True
        self.car_controls.manual_gear = 1       # 전진 기어로 초기화
        self.car_controls.brake = 1.0           # 브레이크를 최대로 밟음
        self.client.setCarControls(self.car_controls)   # 차량에 상태를 적용

        # Step 초기화
        self.step_counter = 0

        # 충돌 플래그 초기화
        self.collision_detected = False

        # 직전 시점에서의 위치 초기화
        self.max_dist = 0

        # 초저속 주행 여부 배열 초기화
        self.slow_state_window = []

        time.sleep(0.5)                         # 적용 대기

        return self._get_observation()          # 현재 차량 상태 반환

    def step(self, action_idx):
        # 1개 Step 진행
        self.step_counter += 1

        lidar_data = self.client.getLidarData()
        lidar_detected = self.lidar_processor.detect_obstacle(lidar_data)
        safe_action = self.avoider.choose_safe_action(self.action_space, lidar_detected)
        if safe_action is not None:
            action_idx = safe_action

        # 행동 적용
        throttle, steering, gear = self.action_space[action_idx]    # Action space에서 행동 선택
        self.car_controls.throttle = throttle
        self.car_controls.steering = steering
        if self.car_controls.manual_gear != gear:
            self.car_controls.is_manual_gear = True
            self.car_controls.manual_gear = gear
        else:
            self.car_controls.is_manual_gear = False
        self.car_controls.brake = 0.0
        self.client.setCarControls(self.car_controls)

        time.sleep(0.1)                         # 적용 대기

        obs = self._get_observation()           # 현재 차량 상태 반환
        
        reward_detail = self._compute_reward()         # 주행 성과에 따른 보상 계산
        
        total_reward = sum(reward_detail.values())

        done_info = self._check_done()          # 종료 조건 계산
        done = done_info['done']
        info = {                                # 디버깅용 보조 정보. 필요에 따라 추가. 학습에 이용되지 않음
            'step': self.step_counter,
            'reward_detail': reward_detail,
            'done_reason': [key for key in ['collision', 'out_of_bounds', 'too_slow', 'timeout'] if done_info[key]]
        }

        # 충돌로 인한 종료시 큰 페널티 부여
        if 'collision' in info['done_reason']:
            total_reward -= 1000

        if done:
            self.car_controls.throttle = 0.0
            self.car_controls.brake = 1.0
            self.client.setCarControls(self.car_controls)                               

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
        
        responses = self.client.simGetImages([
            airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False)
        ], vehicle_name='Car1')
        lidar_data = self.client.getLidarData(lidar_name='LidarSensor')
        
        # 도로 여부에 따른 페널티
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        image = Image.fromarray(img_rgb)
        if self.road_detector.is_on_road(image):
            offroad_penalty = 0.0
        else:
            offroad_penalty = -5.0

        # 센서 기반 페널티
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        if self.avoider.get_min_distance(points) < 1.5:
            sensor_penalty = -100.0
        elif self.avoider.get_min_distance(points) < 3.0:
            sensor_penalty = -10.0
        else:
            sensor_penalty = 0.0

        # 속도 보상
        speed_reward = (-abs(speed - self.target_speed) + self.target_speed) * 5   # 목표 속도에 가까울수록 더 큰 보상 부여. target_speed가 최대 보상

        # 속도 페널티
        low_speed_penalty = 0.0
        if speed < 5.0:
            low_speed_penalty = -10.0

        # 이동 거리 보상
        dx = pos.x_val - self.init_x
        dy = pos.y_val - self.init_y
        dz = pos.z_val - self.init_z

        step_distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if step_distance > self.max_dist:
            distance_reward = step_distance - self.max_dist
            self.max_dist = distance_reward
        else:
            distance_reward = 0.0

        # 후진 페널티
        if self.car_controls.manual_gear == -1:
            reverse_penalty = -10.0
        else:
            reverse_penalty = 0

        # 급회전 페널티
        steering_penalty = -abs(self.car_controls.steering)    # 핸들 회전이 클수록 페널티 부여

        # 도로 이탈 페널티(미구현)
        boundary_penalty = 0.0

        return {
            'offroad_penalty': offroad_penalty,
            'sensor_penalty': sensor_penalty,
            'speed_reward': speed_reward,
            'low_speed_penalty': low_speed_penalty,
            'distance_reward': distance_reward,
            'reverse_penalty': reverse_penalty,
            'steering_penalty': steering_penalty,
            'boundary_penalty': boundary_penalty
        }
    
    def _check_done(self):
        # 충돌 시 종료
        collision_info = self.client.simGetCollisionInfo()
        collision = collision_info.has_collided
        excluded_keywords = []
        if collision:
            if not any(k in collision_info.object_name.lower() for k in excluded_keywords):
                self.collision_detected = True
            else:
                self.collision_detected = False
                collision = False

        # 차량 위치가 한계 범위를 벗어나면 종료
        pos = self.client.getCarState().kinematics_estimated.position
        out_of_bounds = abs(pos.x_val) > self.max_x_val or abs(pos.y_val) > self.max_y_val

        # 차량이 초저속으로 오래 지속되면 종료
        speed = self.client.getCarState().speed
        self.slow_state_window.append(speed < self.low_speed_limit)
        if len(self.slow_state_window) > self.window_n:
            self.slow_state_window.pop(0)
        
        # Step 500 이전에는 저속에 의한 종료 비활성화
        if self.step_counter < 500:
            too_slow = False
        else:
            too_slow =  sum(self.slow_state_window) / len(self.slow_state_window) >= 0.9

        # 최대 step에 도달하면 종료
        timeout = self.step_counter >= self.max_step

        # 종료 조건 확인
        done = collision or out_of_bounds or too_slow or timeout

        # 충돌 플래그 초기화
        if done:
            self.collision_detected = False

        return {
            'done': done,
            'collision': collision,
            'out_of_bounds': out_of_bounds,
            'too_slow': too_slow,
            'timeout': timeout
        }