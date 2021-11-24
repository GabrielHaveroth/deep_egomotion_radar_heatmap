from helpers import *
from dataset_loaders import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_test_data = pd.read_pickle('./metadata/test.pkl')
valid_pairs = []
names = list(df_test_data['file'].unique())
calib_path = '/mnt/Share_Data/Conjuntos_Dados_Mestrado/calib'
all_radar_params = get_cascade_params(calib_path)
poses_seqs = {}

# Loading results
with open('poses.npy', 'rb') as f:
    poses_pred = np.load(f)
with open('angles.npy', 'rb') as f:
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
        if last_pose_idx == pair[0]:
            valid_pairs.append(pair)
            last_pose_idx = pair[1]

poses = poses_seqs[file]
T_p = np.identity(4)
abs_poses = []
abs_poses_pred = []
poses_all = []
for idx, pair in enumerate(valid_pairs):
    t1_idx = pair[0]
    t2_idx = pair[1]
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
# Plot results predict
# plt.plot(abs_poses[:, 0], abs_poses_pred[:, 1])
# plt.plot(abs_poses[:, 0], abs_poses[:, 1])

plt.plot(poses_all[:, 5])
plt.plot(poses_pred[:, 2])
plt.show()