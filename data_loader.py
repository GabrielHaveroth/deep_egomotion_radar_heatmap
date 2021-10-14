import tensorflow as tf
from tensorflow import keras
import numpy as np
from dataset_loaders import *
from helpers import *
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from geometric_operations import *
import pandas as pd
MAX_HEATMAP = np.array([MAX_POWER, MAX_DOPPLER])
MIN_HEATMAP = np.array([MIN_POWER, MIN_DOPPLER])


class RadarEgomotionDataGenerator(keras.utils.Sequence):
    def __init__(self, df_data, params, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.df_data = df_data
        self.shuffle = shuffle
        self.gt_params = get_groundtruth_params()
        self.heat_map_params = params['heatmap']
        self.n = len(df_data)

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
        poses = dict()
        for name in names:
            gt_poses = get_groundtruth(name)
            radar_timestamps = get_timestamps(name, self.heat_map_params)
            gt_timestamps = get_timestamps(name, self.gt_params)
            radar_gt, _ = interpolate_poses(gt_poses,
                                            gt_timestamps,
                                            radar_timestamps)
            poses[name] = radar_gt
        return poses

    def __get_data(self, batches):
        names = list(batches['file'].unique())
        # Generates data containing batch_size samples
        poses_per_name = self.__get_gt_pose_by_file(names)
        cart_heatmapst12 = []
        elev_radar_heatmapt12 = []
        delta_trans = []
        delta_rot = []
        for _, row in batches.iterrows():
            seq = row['file']
            pair = row['pair']
            poses = poses_per_name[seq]
            radar_heatmap_t_1 = get_heatmap(pair[0], seq,  self.heat_map_params)
            radar_heatmap_t_2 = get_heatmap(pair[1], seq,  self.heat_map_params)
            cart_radar_heatmap_t_1 = (get_cartesian_slice_heatmap(radar_heatmap_t_1) - MIN_HEATMAP) / (MAX_HEATMAP - MIN_HEATMAP)
            cart_radar_heatmap_t_2 = (get_cartesian_slice_heatmap(radar_heatmap_t_2) - MIN_HEATMAP) / (MAX_HEATMAP - MIN_HEATMAP)
            elev_radar_heatmap_t_1 = (get_range_elevation_slice_heatmap(radar_heatmap_t_1) - MIN_HEATMAP) / (MAX_HEATMAP - MIN_HEATMAP)
            elev_radar_heatmap_t_2 = (get_range_elevation_slice_heatmap(radar_heatmap_t_2) - MIN_HEATMAP) / (MAX_HEATMAP - MIN_HEATMAP)
            cart_heatmapst12.append(np.concatenate([cart_radar_heatmap_t_1, cart_radar_heatmap_t_2], -1))
            elev_radar_heatmapt12.append(np.concatenate([elev_radar_heatmap_t_1, elev_radar_heatmap_t_2], -1))
            delta_pose, _ = get_ground_6d_poses_euler(poses[pair[0]], poses[pair[1]])
            delta_rot.append(delta_pose[0:3])
            delta_trans.append(delta_pose[3:])
        X_batch_cart = np.asarray(cart_heatmapst12)
        X_batch_elev = np.asarray(elev_radar_heatmapt12)
        y_batch_trans = np.asarray(delta_trans)
        y_batch_rot = np.asarray(delta_rot)
        return [X_batch_cart, X_batch_elev], [y_batch_trans, y_batch_rot]
