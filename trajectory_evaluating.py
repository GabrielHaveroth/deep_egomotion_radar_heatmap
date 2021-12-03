from math import sin
from numpy.lib.function_base import append
from helpers import *
from dataset_loaders import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_test_data = pd.read_pickle('./metadata/test.pkl')
valid_pairs = []
names = list(df_test_data['file'].unique())
calib_path = '/home/lactec/dados/mestrado_gabriel/calib/'
all_radar_params = get_cascade_params(calib_path)
poses_seqs = {}


index = df_test_data.index
number_of_rows = len(index)
print(number_of_rows)
_POSE_6D = False
# Loading results
with open('./results/poses.npy', 'rb') as f:
    poses_pred = np.load(f)
with open('./results/angles.npy', 'rb') as f:
    angles_pred = np.load(f)

# Loading GT
for name in names:
    poses_seqs[name] = get_heatmap_poses(name, all_radar_params['heatmap'])

for idx, df_row in enumerate(df_test_data.iterrows()):
    pair = df_row[1]['heatmap_pairs']
    file = df_row[1]['file']
    if idx == 0:
        last_pose_idx = pair[1]
        valid_pairs.append(pair)
    else:
        df_test_data

T_p = np.identity(4)
abs_poses = []
abs_poses_pred = []
poses_all = []
if _POSE_6D:
    for idx, df_row in enumerate(df_test_data.iterrows()):
        pair = df_row[1]['heatmap_pairs']
        file = df_row[1]['file']
        t1_idx = pair[0]
        t2_idx = pair[1]
        poses = poses_seqs[file]
        pose_t1 = poses[t1_idx]
        pose_t2 = poses[t2_idx]
        delta_pose, T_p_12 = get_ground_6d_poses_euler(pose_t1,
                                                       pose_t2)
        T_p = T_p.dot(T_p_12)
        poses_all.append(delta_pose)
        abs_poses.append(get_6D_poses_from_matrix(T_p)[3:])
    abs_poses = np.array(abs_poses)
    poses_all = np.array(poses_all)
    T_p_pred = np.identity(4)
    # Plot GT
    for delta_pose, delta_angle in zip(poses_pred, angles_pred):
        T_p_12 = get_matrix_from_6D_relative_pose(delta_angle, delta_pose)
        T_p_pred = T_p_pred.dot(T_p_12)
        abs_poses_pred.append(get_6D_poses_from_matrix(T_p_pred)[3:])
    abs_poses_pred = np.array(abs_poses_pred)
# 2D pose type (x, y, theta)
else:
    x_gt_abs = []
    y_gt_abs = []
    x_gt = 0
    y_gt = 0
    delta_theta = 0
    T_p = np.identity(4)
    delta_poses_gt = []
    abs_poses = [np.zeros((6))] # zero as init position
    for idx, df_row in enumerate(df_test_data.iterrows()):
        pair = df_row[1]['heatmap_pairs']
        file = df_row[1]['file']
        t1_idx = pair[0]
        t2_idx = pair[1]
        poses = poses_seqs[file]
        pose_t1 = poses[t1_idx]
        pose_t2 = poses[t2_idx]
        delta_pose, T_p_12 = get_ground_6d_poses_euler(pose_t1,
                                                       pose_t2)

        T_p = T_p.dot(T_p_12)
        delta_x_gt = delta_pose[4]
        delta_y_gt = delta_pose[5]
        poses_all.append(delta_pose)
        y_gt_abs.append(y_gt)
        delta_poses_gt.append(delta_pose)
        poses_all.append(delta_pose)
        abs_poses.append(get_6D_poses_from_matrix(T_p))

    abs_poses = np.array(abs_poses)
    delta_poses_gt = abs_poses[1:, 3:5] - abs_poses[0:-1, 3:5]
    delta_thetas_gt = abs_poses[1:, 2] - abs_poses[0:-1, 2]

    x_gt = 0
    y_gt = 0
    x_gt_abs = [0.0]
    y_gt_abs = [0.0]
    delta_theta = 0
    for delta_pose_gt in delta_poses_gt:
        delta_l = np.sqrt(delta_pose_gt[0] ** 2 + delta_pose_gt[1] ** 2)
        delta_theta = delta_theta + np.arctan(delta_pose_gt[1]  / delta_pose_gt[0])
        if delta_theta < -np.pi:
            delta_theta += 2 * np.pi
        elif delta_theta > np.pi:
            delta_theta -= 2 * np.pi
        # x_gt = x_gt + delta_l * np.cos(delta_theta)
        # y_gt = y_gt + delta_l * np.sin(delta_theta)
        x_gt = x_gt + delta_pose_gt[0]
        y_gt = y_gt + delta_pose_gt[1]
        x_gt_abs.append(x_gt)
        y_gt_abs.append(y_gt)


    poses_all = np.array(poses_all)
    x_gt_abs = np.array(x_gt_abs)
    y_gt_abs = np.array(y_gt_abs)
    # Plot GT
    # for delta_pose, delta_angle in zip(poses_pred, angles_pred):
    #     T_p_12 = get_matrix_from_6D_relative_pose(delta_angle, delta_pose)
    #     T_p_pred = T_p_pred.dot(T_p_12)
    #     abs_poses_pred.append(get_6D_poses_from_matrix(T_p_pred)[3:])
    # abs_poses_pred = np.array(abs_poses_pred)

# Plot results predict
# plt.plot(abs_poses[:, 0], abs_poses_pred[:, 1])
# plt.plot(abs_poses[:, 0], abs_poses[:, 1])

# plt.plot(abs_poses[:, 4], abs_poses[:, 3])
plt.plot(y_gt_abs, x_gt_abs)

plt.show()
