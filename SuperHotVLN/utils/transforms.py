import numpy as np

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def transform_to_sim_position(xyz_position):
    """
    Transforms position by swapping y and z axes since Isaac Sim uses z as vertical while R2R uses y as vertical.
    """
    return [xyz_position[0], xyz_position[2], xyz_position[1]]

def transform_to_sim_rotation(quaternion_rotation):
    """
    Transforms rotation by swapping y and z axes since Isaac Sim uses z as vertical while R2R uses y as vertical.
    """
    return [quaternion_rotation[3], quaternion_rotation[0], quaternion_rotation[2], quaternion_rotation[1]]


