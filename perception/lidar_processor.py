import numpy as np

class LidarProcessor:
    def __init__(self, min_distance=2.0):
        self.min_distance = min_distance
    
    def detect_obstacle(self, lidar_data):
        if not lidar_data.point_cloud:
            return False
        
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        distances = np.linalg.norm(points[:, :2], axis=1)
        return np.any(distances < self.min_distance)