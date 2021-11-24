from helpers import interpolate_poses
from dataset_loaders import *
import numpy as np
import pandas as pd


def get_closest_value(arr: np.array, value) -> int:
    idx = (np.abs(arr - value)).argmin()
    return idx

# Put her seqs to generate train metadata
TRAIN_SEQS = ["12_21_2020_ec_hallways_run0",
              "12_21_2020_ec_hallways_run2",
              "12_21_2020_ec_hallways_run3",
              "12_21_2020_ec_hallways_run4",
              "12_21_2020_arpg_lab_run0",
              "12_21_2020_arpg_lab_run1",
              "12_21_2020_arpg_lab_run2",
              "12_21_2020_arpg_lab_run3",
              "2_23_2021_edgar_classroom_run0",
              "2_23_2021_edgar_classroom_run1",
              "2_23_2021_edgar_classroom_run3",
              "2_23_2021_edgar_classroom_run4",
              "2_23_2021_edgar_classroom_run5",
              "2_22_2021_longboard_run2",
              "2_22_2021_longboard_run3",
              "2_22_2021_longboard_run4",
              "2_22_2021_longboard_run5",
              "2_22_2021_longboard_run6",
              "2_22_2021_longboard_run7",
              "2_28_2021_outdoors_run0",
              "2_28_2021_outdoors_run1",
              "2_28_2021_outdoors_run2",
              "2_28_2021_outdoors_run3",
              "2_28_2021_outdoors_run4",
              "2_28_2021_outdoors_run7",
              "2_28_2021_outdoors_run8",
              "2_28_2021_outdoors_run9"]
calib_path = '/home/lactec/dados/mestrado_gabriel/calib'
seqs = TRAIN_SEQS
# Path to dataset
path_data = '/home/lactec/dados/mestrado_gabriel/coloradar/'
file_name_metadata = '/home/lactec/Codigos_Mestrado_GabrielH/deep_egomotion_radar_heatmap/train'
all_radar_params = get_cascade_params(calib_path)
radar_heatmap_params = all_radar_params['heatmap']

MIN_TIME_BETWEEN_PAIR = 0.4
MAX_TIME_BETWEEN_PAIR = 1
TRAIN_PERCENTAGE = 1

init_time_stamp = 0
unique_pair = True
data = {}
pairs = []
files = []
imu_points = []
neast_imu_points = {}

for seq in seqs:
    name = path_data + seq
    gt_params = get_groundtruth_params()
    radar_timestamps = get_timestamps(name, radar_heatmap_params)
    gt_timestamps = get_timestamps(name, gt_params)
    imu_data = get_imu(name)
    imu_params = get_imu_params(calib_path)
    imu_timestamps = np.asarray(get_timestamps(name, imu_params))
    # Get groundtruth poses
    gt_poses = get_groundtruth(name)
    # Interpolate groundtruth poses for each sensor measurement
    radar_gt, radar_indices = interpolate_poses(gt_poses,
                                                gt_timestamps,
                                                radar_timestamps)
    # Select radar timestamps
    selected_timestamps_radar = [radar_timestamps[idx]
                                 for idx in radar_indices]
    # Take neast imu points to heatmaps timestamp
    for idx, radar_timestamp in enumerate(selected_timestamps_radar):
        neast_imu_points[radar_indices[idx]] = get_closest_value(imu_timestamps, radar_timestamp)

    for i in range(len(selected_timestamps_radar)):
        for j in range(i + 1, len(selected_timestamps_radar)):
            timestamp_diff = round(
                selected_timestamps_radar[j] - selected_timestamps_radar[i], 2)
            if MIN_TIME_BETWEEN_PAIR <= timestamp_diff <= MAX_TIME_BETWEEN_PAIR:
                pairs.append((radar_indices[i], radar_indices[j]))
                files.append(name)
                imu_points.append((neast_imu_points[radar_indices[i]], 
                                   neast_imu_points[radar_indices[j]]))
                if unique_pair:
                    break

data['heatmap_pairs'] = pairs
data['file'] = files
data['imu_pairs'] = imu_points
df_data = pd.DataFrame(data)
df_data.to_pickle(file_name_metadata + '.pkl')
print(df_data)
