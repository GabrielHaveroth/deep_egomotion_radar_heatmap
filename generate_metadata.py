from dataset_loaders import *
from helpers import *
import pandas as pd


def get_closest_value(arr: np.array, value) -> int:
    idx = (np.abs(arr - value)).argmin()
    return idx

# Loading dataset
calib_path = '/mnt/Share_Data/Conjuntos_Dados_Mestrado/calib'
seqs = ['2_28_2021_outdoors_run1']
path = '/mnt/Share_Data/Conjuntos_Dados_Mestrado/'
file_name_metadata = './models/test'

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
imu_data = []
neast_imu_data = {}

for seq in seqs:
    name = path + seq
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
        neast_imu_data[radar_indices[idx]] = get_closest_value(
            imu_timestamps, radar_timestamp)

    for i in range(len(selected_timestamps_radar)):
        for j in range(i + 1, len(selected_timestamps_radar)):
            timestamp_diff = round(
                selected_timestamps_radar[j] - selected_timestamps_radar[i], 2)
            if MIN_TIME_BETWEEN_PAIR <= timestamp_diff <= MAX_TIME_BETWEEN_PAIR:
                pairs.append((radar_indices[i], radar_indices[j]))
                files.append(name)
                imu_data.append(neast_imu_data[radar_indices[j]])
                if unique_pair:
                    break

valid_pairs = []
valid_files = []
valid_imu_data = []
for idx, pair in enumerate(pairs):
    if idx == 0:
        last_pose_idx = pair[1]
        valid_pairs.append(pair)
        valid_files.append(files[idx])
        valid_imu_data.append(imu_data[idx])
    else:
        if last_pose_idx == pair[0]:
            valid_pairs.append(pair)
            valid_files.append(files[idx])
            valid_imu_data.append(imu_data[idx])
            last_pose_idx = pair[1]

pairs = valid_pairs
files = valid_files
# imu_data = valid_imu_data
data['heatmap_pairs'] = pairs
data['file'] = files
# data['imu_data'] = imu_data
df_data = pd.DataFrame(data)
df_data.to_pickle(file_name_metadata + '.pkl')
print(df_data)