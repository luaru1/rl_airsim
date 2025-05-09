import numpy as np

class ObstacleAvoider:
    def __init__(self):
        pass

    def get_min_distance(self, point_cloud):
        if point_cloud.size == 0:
            return float('inf')
        distances = np.linalg.norm(point_cloud, axis=1)
        return distances.min()

    def choose_safe_action(self, action_space, lidar_detected):
        if not lidar_detected:
            return None
        
        for i, (throttle, steering, gear) in enumerate(action_space):
            if abs(steering) > 0.4:
                return i
        return None