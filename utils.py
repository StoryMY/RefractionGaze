import numpy as np

def pitchyaw_to_unit_vector(pitchyaw):
    """
    cosy 0 -siny | 1   0   0     |  0
      0  1  0    | 0 cosp sinp   |  0
    siny 0 cosy  | 0 -sinp cosp  |  1
    """
    pitch = pitchyaw[0]
    yaw = pitchyaw[1]
    x = -np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return np.array([x, y, z])

def compute_angle_error(pred, gt):
    pred_vec = pitchyaw_to_unit_vector(pred)
    gt_vec = pitchyaw_to_unit_vector(gt)
    dot = np.dot(pred_vec, gt_vec)
    angles = dot / np.linalg.norm(pred_vec) / np.linalg.norm(gt_vec)
    return np.arccos(angles) #* 180 / np.pi
