from dataset_loaders import *
from helpers import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

def get_closest_value(arr: np.array, value) -> int:
    idx = (np.abs(arr - value)).argmin()
    return idx

# Options
USE_SCALER = True
TYPE = "TEST"
MIN_TIME_BETWEEN_PAIR = 0.4
MAX_TIME_BETWEEN_PAIR = 1
UNIQUE_PAIR = True
USE_TEMPORAL_ORDER = True


# Put her seqs to generate train metadata

TRAIN_SEQS = ["12_21_2020_ec_hallways_run0",
              "12_21_2020_ec_hallways_run2",
              "12_21_2020_ec_hallways_run3",
              "12_21_2020_ec_hallways_run4",
              "12_21_2020_arpg_lab_run0",
              "12_21_2020_arpg_lab_run1",
              "12_21_2020_arpg_lab_run2",
              "12_21_2020_arpg_lab_run3",
              "2_28_2021_outdoors_run0",
              "2_28_2021_outdoors_run1",
              "2_28_2021_outdoors_run2",
              "2_28_2021_outdoors_run3",
              "2_28_2021_outdoors_run4",
              "2_28_2021_outdoors_run7",
              "2_28_2021_outdoors_run8",
              "2_28_2021_outdoors_run9",
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
              "2_22_2021_longboard_run7"]

# Put her seqs to generate TEST metadata
TEST_SEQS = ["2_28_2021_outdoors_run5"]


if TYPE == 'TRAIN':
    seqs = TRAIN_SEQS
    file_name_metadata = '/home/lactec/Codigos_Mestrado_GabrielH/deep_egomotion_radar_heatmap/metadata/train'
    
elif TYPE == 'TEST':
    seqs = TEST_SEQS
    file_name_metadata = '/home/lactec/Codigos_Mestrado_GabrielH/deep_egomotion_radar_heatmap/metadata/test'

# Path to dataset
path_data = '/home/lactec/dados/mestrado_gabriel/coloradar/'
# Loading dataset
calib_path = '/home/lactec/dados/mestrado_gabriel/calib'
all_radar_params = get_cascade_params(calib_path)
radar_heatmap_params = all_radar_params['heatmap']


imu_pairs = []
all_files = []
all_pairs = []
all_imu_pairs = []
data = {}
neast_imu_data = {}
all_delta_poses_2D = []

for seq in seqs:
    pairs = []
    files = []
    imu_pairs = []
    delta_poses_2D = []
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
        neast_imu_data[idx] = get_closest_value(imu_timestamps, radar_timestamp)

    for i in range(len(selected_timestamps_radar)):
        for j in range(i + 1, len(selected_timestamps_radar)):
            timestamp_diff = round(selected_timestamps_radar[j] - selected_timestamps_radar[i], 2)
            if MIN_TIME_BETWEEN_PAIR <= timestamp_diff <= MAX_TIME_BETWEEN_PAIR:
                pairs.append((i, j))
                imu_pairs.append((neast_imu_data[i], neast_imu_data[j]))
                files.append(name)
                if UNIQUE_PAIR:
                    break
                
    if USE_TEMPORAL_ORDER:            
        valid_pairs = []
        valid_files = []
        valid_imu_pairs = []
        for idx, pair in enumerate(pairs):
            if idx == 0:
                last_pose_idx = pair[1]
                valid_pairs.append(pair)
                valid_files.append(files[idx])
                valid_imu_pairs.append(imu_pairs[idx])
            else:
                if last_pose_idx == pair[0]:
                    valid_pairs.append(pair)
                    valid_files.append(files[idx])
                    valid_imu_pairs.append(imu_pairs[idx])
                    last_pose_idx = pair[1]
        # Replace for just valid pairs for test [(0, 1), (1, 2), ...]           
        pairs = valid_pairs
        files = valid_files
        imu_pairs = valid_imu_pairs
        delta_poses_2D = get_2D_delta_poses(pairs, radar_gt)
    all_files = all_files + files
    all_pairs = all_pairs + pairs
    all_imu_pairs = all_imu_pairs + imu_pairs
    all_delta_poses_2D = all_delta_poses_2D + delta_poses_2D.tolist()

if TYPE == 'TRAIN':
    scaler = MinMaxScaler(np.array(all_delta_poses_2D))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))

data['heatmap_pairs'] = all_pairs
data['file'] = all_files
data['imu_data'] = all_imu_pairs
data['delta_poses_2D'] = all_delta_poses_2D
df_data = pd.DataFrame(data)
delta_poses = df_data['delta_poses_2D'].values.copy()
delta_poses_arr = []
for delta_p in delta_poses:
    delta_p = np.array(delta_p)
    delta_poses_arr.append(delta_p)    
delta_poses_arr = np.array(delta_poses_arr)
print(delta_poses_arr)
df_data.to_pickle(file_name_metadata + '.pkl')
