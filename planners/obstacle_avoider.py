class ObstacleAvoider:
    def __init__(self):
        pass

    def choose_safe_action(self, action_space, lidar_detected):
        if not lidar_detected:
            return None
        
        for i, (throttle, steering, gear) in enumerate(action_space):
            if abs(steering) > 0.4:
                return i
        return None