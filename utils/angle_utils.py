import math

def quaternion_to_euler(q):
    # 쿼터니언 값
    x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val

    # Roll (x축 회전)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y축 회전)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, sinp))

    # Yaw (z축 회전)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def radians_to_degrees(r):
    return math.degrees(r)

def degrees_to_radians(d):
    return math.radians(d)