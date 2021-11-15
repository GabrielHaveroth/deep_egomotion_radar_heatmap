from tensorflow.python.keras.regularizers import get
from data_loader import *
from dataset_loaders import *


def get_closest_value(arr: np.array, value) -> int:
    idx = (np.abs(arr - value)).argmin()
    return idx

# Loading dataset
calib_path = '/data/Conjuntos_Dados_Mestrado/calib'
seqs = ['2_22_2021_longboard_run7/', '2_23_2021_edgar_classroom_run5/']
path = '/data/Conjuntos_Dados_Mestrado/'
file_name_metadata = './models/train'

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
        neast_imu_points[radar_indices[idx]] = get_closest_value(imu_timestamps, radar_timestamp)

    for i in range(len(selected_timestamps_radar)):
        for j in range(i + 1, len(selected_timestamps_radar)):
            timestamp_diff = round(
                selected_timestamps_radar[j] - selected_timestamps_radar[i], 2)
            if MIN_TIME_BETWEEN_PAIR <= timestamp_diff <= MAX_TIME_BETWEEN_PAIR:
                pairs.append((radar_indices[i], radar_indices[j]))
                files.append(name)
                imu_points.append(neast_imu_points[radar_indices[j]])
                if unique_pair:
                    break

data['heatmap_pair'] = pairs
data['file'] = files
data['imu_points'] = imu_points
df_data = pd.DataFrame(data)
df_data.to_pickle(file_name_metadata + '.pkl')

