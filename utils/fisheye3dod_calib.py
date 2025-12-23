import numpy as np
import re
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ocamcamera import OcamCamera


def get_matrix(location, rotation):
    """
    Creates matrix from carla transform.
    """
    x, y, z = location[0], location[1], location[2]
    roll, pitch, yaw = rotation[0], rotation[1], rotation[2]

    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.eye(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def get_camera_intrinsics(img_height, img_width, view_fov):
    calibration = np.eye(4, dtype=np.float32)
    calibration[0, 2] = img_width / 2.0
    calibration[1, 2] = img_height / 2.0
    calibration[0, 0] = img_width / (2.0 * np.tan(view_fov * np.pi / 360.0))
    calibration[1, 1] = calibration[0, 0]
    return calibration

def lidar_coord_to_cam_coord(matrix):
    #           ^ z                . z
    #           |                 /
    #           |       to:      +-------> x
    #           |  . x           |
    #           | /              |
    # y <-------+                v y
    M = np.array([[ 0, -1,  0,  0 ],
                  [ 0,  0, -1,  0 ],
                  [ 1,  0,  0,  0 ],
                  [ 0,  0,  0,  1 ]])
    return np.dot(M, matrix) # first transform to camera position, then to axis of camera coord

def cam_coord_to_lidar_coord(matrix):
    #    . z                      ^ z
    #   /                         |
    #  +-------> x  to:           |
    #  |                          |  . x
    #  |                          | /
    #  v y              y <-------+
    M = np.array([[ 0,  0,  1,  0 ],
                  [-1,  0,  0,  0 ],
                  [ 0, -1,  0,  0 ],
                  [ 0,  0,  0,  1 ]])
    return np.dot(matrix, M)  # first transform to axis of lidar coord, then to lidar position


def parse_sensor_transform(filename):
    sensor_transforms = {}
    pattern = re.compile(r"""
        frame=(\d+),\s+
        .*?
        Transform\(
            Location\(x=([-.\d]+),\s+y=([-.\d]+),\s+z=([-.\d]+)\),\s+
            Rotation\(pitch=([-.\d]+),\s+yaw=([-.\d]+),\s+roll=([-.\d]+)\)
        \)
    """, re.X)

    with open(filename, 'r') as f:
        data = f.read()
        # data = f.readline()  # only first line
        matches = pattern.findall(data)
        for match in matches:
            frame_id = match[0].zfill(8)
            x = float(match[1])
            y = -float(match[2])
            z = float(match[3])
            location = [x, y, z]  # transform carla coord to lidar coord system, x, -y, z
            pitch = -float(match[4])
            yaw = -float(match[5])
            roll = float(match[6])
            # transform carla coord to lidar coord system, roll, -pitch, -yaw
            rotation = [roll, pitch, yaw]

            sensor_transforms[frame_id] = {
                'location': location,
                'rotation': rotation
            }
    return sensor_transforms


class Fisheye3DODCalib:
    VIEW_FOV = {
        'cam_rgb': 98.5, 'cam_nusc': 90, 'cam_fisheye': 220, 'cam_dvs': 104.7
    }
    def __init__(self, cam_type, sensor_transform, lidar_transform, img_height, img_width, ocam_path=None):
        sensor2world = get_matrix(
            sensor_transform['location'], sensor_transform['rotation'])
        lidar2world = get_matrix(
            lidar_transform['location'], lidar_transform['rotation'])
        world2sensor = np.linalg.inv(sensor2world)
        world2lidar = np.linalg.inv(lidar2world)
        sensor2lidar = np.dot(world2lidar, sensor2world)
        lidar2sensor = np.dot(world2sensor, lidar2world)

        self.cam2lidar = cam_coord_to_lidar_coord(sensor2lidar)
        self.lidar2cam = lidar_coord_to_cam_coord(lidar2sensor)
        
        if cam_type != 'cam_fisheye':
            self.cam2img = get_camera_intrinsics(
                img_height, img_width, self.VIEW_FOV[cam_type])
            self.lidar2img = np.dot(self.cam2img, self.lidar2cam)
        else:
            self.cam2img = OcamCamera(
                filename=ocam_path, fov=self.VIEW_FOV[cam_type])
            self.lidar2img = np.eye(4)