import json
import numpy as np
import cv2
from . import d3

skel_girl_joints = [
    'pelvis',
    'spine_01',
    'spine_02',
    'spine_03',
    'clavicle_l',
    'upperarm_l',
    'lowerarm_l',
    'hand_l',
    'clavicle_r',
    'upperarm_r',
    'lowerarm_r',
    'hand_r',
    'neck_01',
    'head',
    'thigh_l',
    'calf_l',
    'foot_l',
    'ball_l',
    'thigh_r',
    'calf_r',
    'foot_r',
    'ball_r',
]

def load_2d_kps(kp_path, joints_name=None):
    joints_name = skel_girl_joints
    kp_3d = json.load(open(kp_path))
    kpts_3d_array = []
    kpts_3d = {}
    for kp in kp_3d['kp']:
        if kp['Name'].lower() in joints_name:
            kpts_3d[kp['Name'].lower()] = [kp['KpWorld']['X'], kp['KpWorld']['Y'], kp['KpWorld']['Z']]
    kpts_3d_array = np.array([kpts_3d[k] for k in joints_name])
    width = 640
    height = 480
    x, y, z = kp_3d['cam_loc']
    pitch, yaw, roll = kp_3d['cam_rot']
    cam_pose = d3.CameraPose(x, y, z, pitch, yaw, roll, width, height, width / 2)

    # points_2d = cam_pose.project_to_2d(points_3d)  # list of 2d point, x, y
    points_3d_cam = cam_pose.project_to_cam_space(kpts_3d_array)
    kpts_2d = points_3d_cam[:,:2]
    return kpts_2d

def project_to_img(kpts_2d, img):
    img = img.copy()
    for kp in kpts_2d:
        x = kp[0]
        y = kp[1]
        cv2.circle(img, (int(x),int(y)), radius=2, color=(0,255,0), thickness=2)
    return img
