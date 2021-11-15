import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.backend import set_learning_phase
from dataset_loaders import *
from helpers import *
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from geometric_operations import *
import pandas as pd
import multiprocessing as mp
MAX_HEATMAP = np.array([MAX_POWER, MAX_DOPPLER])
MIN_HEATMAP = np.array([MIN_POWER, MIN_DOPPLER])


class RadarEgomotionDataGenerator(keras.utils.Sequence):
    def __init__(self, df_data, params, batch_size=32, shuffle=True,
                 data_type="slices"):
        self.batch_size = batch_size
        self.df_data = df_data
        self.shuffle = shuffle
        self.gt_params = get_groundtruth_params()
        self.hm_params = params['heatmap']
        self.n = len(df_data)
        self.data_type = data_type

    def __len__(self):
        return int(np.floor(self.n / self.batch_size))

    def __getitem__(self, index):

        batches = self.df_data[index *
                               self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.df_data = self.df_data.sample(frac=1).reset_index(drop=True)

    def __get_gt_pose_by_file(self, names):
        poses_seq = dict()
        with mp.Pool(mp.cpu_count()) as pool:
            results = [pool.apply_async(get_heatmap_poses, args=(
                name, self.hm_params)) for name in names]
            for r, name in zip(results, names):
                poses_seq[name] = r.get()
        return poses_seq

    def __get_data(self, batches: pd.DataFrame):
        names = list(batches['file'].unique())
        # Generates data containing batch_size samples
        poses_seq = self.__get_gt_pose_by_file(names)
        delta_trans = []
        delta_rot = []
        if self.data_type == 'custom':
            cart_heatmapst12 = []
            elev_radar_heatmapt12 = []
            for _, row in batches.iterrows():
                seq = row['file']
                pair = row['pair']
                poses = poses_seq[seq]
                radar_heatmap_t_1 = get_heatmap(
                    pair[0], seq,  self.heat_map_params)
                radar_heatmap_t_2 = get_heatmap(
                    pair[1], seq,  self.heat_map_params)
                cart_radar_heatmap_t_1 = (get_cartesian_slice_heatmap(
                    radar_heatmap_t_1) - MIN_HEATMAP) / (MAX_HEATMAP - MIN_HEATMAP)
                cart_radar_heatmap_t_2 = (get_cartesian_slice_heatmap(
                    radar_heatmap_t_2) - MIN_HEATMAP) / (MAX_HEATMAP - MIN_HEATMAP)
                elev_radar_heatmap_t_1 = (get_range_elevation_slice_heatmap(
                    radar_heatmap_t_1) - MIN_HEATMAP) / (MAX_HEATMAP - MIN_HEATMAP)
                elev_radar_heatmap_t_2 = (get_range_elevation_slice_heatmap(
                    radar_heatmap_t_2) - MIN_HEATMAP) / (MAX_HEATMAP - MIN_HEATMAP)
                cart_heatmapst12.append(np.concatenate(
                    [cart_radar_heatmap_t_1, cart_radar_heatmap_t_2], -1))
                elev_radar_heatmapt12.append(np.concatenate(
                    [elev_radar_heatmap_t_1, elev_radar_heatmap_t_2], -1))
                delta_pose, _ = get_ground_6d_poses_euler(
                    poses[pair[0]], poses[pair[1]])
                delta_rot.append(delta_pose[0:3])
                delta_trans.append(delta_pose[3:])
            X_batch_cart = np.asarray(cart_heatmapst12)
            X_batch_elev = np.asarray(elev_radar_heatmapt12)
            y_batch_trans = np.asarray(delta_trans)
            y_batch_rot = np.asarray(delta_rot)
            data_batch = [[X_batch_cart, X_batch_elev],
                          [y_batch_trans, y_batch_rot]]

        # Get the original 3D power heatmap elev x azim x range
        elif self.data_type == '3d_heatmap':
            delta_poses = []
            hm_powers_t12 = []
            with mp.Pool(mp.cpu_count()) as pool:
                results_async = [pool.apply_async(get_data_3D_heatmap_batch_gt, args=(
                    row, self.hm_params, poses_seq, )) for _, row in batches.iterrows()]
                for result in results_async:
                    hm_power_t12, delta_pose = result.get()
                    delta_poses.append(delta_pose)
                    hm_powers_t12.append(hm_power_t12)
            delta_poses = np.asarray(delta_poses)
            y_batch_trans = delta_poses[:, 3:].copy()
            y_batch_rot = delta_poses[:, 0:3].copy()
            X_batch_power_heatmap = np.array(hm_powers_t12)
            data_batch = [X_batch_power_heatmap, [y_batch_trans, y_batch_rot]]

        X_batch = data_batch[0]
        print(type(X_batch))
        y_batch = data_batch[1]
        return X_batch, y_batch
