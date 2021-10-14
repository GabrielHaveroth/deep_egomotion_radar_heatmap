from scipy.spatial.transform import Rotation, Slerp
from geometric_operations import *
from dataset_loaders import *
from helpers import *

seqs = ['/mnt/Share_Data/Conjuntos_Dados_Mestrado/2_23_2021_edgar_classroom_run5/']
max_values_power = []
min_values_power = []
max_values_doppler = []
min_values_doppler = []
all_radar_params = get_cascade_params(
    '/mnt/Share_Data/Conjuntos_Dados_Mestrado/calib')
radar_heatmap_params = all_radar_params['heatmap']
gt_params = get_groundtruth_params()

for seq in seqs:

    radar_timestamps = get_timestamps(seq, radar_heatmap_params)
    gt_timestamps = get_timestamps(seq, gt_params)
    # get groundtruth poses
    gt_poses = get_groundtruth(seq)
    # interpolate groundtruth poses for each sensor measurement
    radar_gt, radar_indices = interpolate_poses(gt_poses,
                                                gt_timestamps,
                                                radar_timestamps)
    for ridx in radar_indices:
        heatmap = get_heatmap(ridx, seq, radar_heatmap_params)
        heatmap_power = heatmap[:, :, :, 0]
        heatmap_doppler = heatmap[:, :, :, 1]
        actual_max_power = heatmap_power.max()
        actual_min_power = heatmap_power.min()
        actual_max_doppler = heatmap_doppler.max()
        actual_min_doppler = heatmap_doppler.min()
        max_values_power.append(actual_max_power)
        min_values_power.append(actual_min_power)
        max_values_doppler.append(actual_max_doppler)
        min_values_doppler.append(actual_min_doppler)

max_all_data_power = np.array(max_values_power).max()
min_all_data_power = np.array(min_values_power).min()
max_all_data_doppler = np.array(max_values_doppler).max()
min_all_data_doppler = np.array(min_values_doppler).min()

print("max power: {}".format(max_all_data_power))
print("min power: {}".format(min_all_data_power))
print("max doppler: {}".format(max_all_data_doppler))
print("min doppler: {}".format(min_all_data_doppler))
