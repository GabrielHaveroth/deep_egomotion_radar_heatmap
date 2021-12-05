from scipy.spatial.transform import Rotation, Slerp
from geometric_operations import *
from dataset_loaders import *
from helpers import *
import multiprocessing as mp
import pandas as pd
from glob import glob

PATH_DATA = '/home/lactec/dados/mestrado_gabriel/'
seqs = glob(PATH_DATA + 'coloradar/*')

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
seqs = TRAIN_SEQS         
max_values_power = []
min_values_power = []
max_values_doppler = []
min_values_doppler = []
all_radar_params = get_cascade_params(PATH_DATA + 'calib')
radar_heatmap_params = all_radar_params['heatmap']
gt_params = get_groundtruth_params()


def get_max_min_hm(ridx, seq, radar_heatmap_params):
    heatmap = get_heatmap(ridx, seq, radar_heatmap_params)
    heatmap_power = heatmap[:, :, :, 0]
    heatmap_doppler = heatmap[:, :, :, 1]
    max_power = heatmap_power.max()
    min_power = heatmap_power.min()
    max_doppler = heatmap_doppler.max()
    min_doppler = heatmap_doppler.min()
    return max_power, min_power, max_doppler, min_doppler

for seq in seqs:
    seq = PATH_DATA + "coloradar/" + seq 
    radar_timestamps = get_timestamps(seq, radar_heatmap_params)
    gt_timestamps = get_timestamps(seq, gt_params)
    # get groundtruth poses
    gt_poses = get_groundtruth(seq)
    # Skip sequences that does't have GT
    if len(gt_poses) == 0:
        print("skip {}".format(seq))
        pass
    else:
        # interpolate groundtruth poses for each sensor measurement
        radar_gt, radar_indices = interpolate_poses(gt_poses,
                                                    gt_timestamps,
                                                    radar_timestamps)                                           
        with mp.Pool(mp.cpu_count()) as pool:
            results_async = [pool.apply_async(get_max_min_hm, args=(ridx, seq, radar_heatmap_params,)) for ridx in radar_indices]
            for r in results_async:
                max_power, min_power, max_doppler, min_doppler = r.get()
                max_values_power.append(max_power)
                min_values_power.append(min_power)
                max_values_doppler.append(max_doppler)
                min_values_doppler.append(min_doppler)

max_all_data_power = np.array(max_values_power).max()
min_all_data_power = np.array(min_values_power).min()
max_all_data_doppler = np.array(max_values_doppler).max()
min_all_data_doppler = np.array(min_values_doppler).min()
data = {'max_power': [max_all_data_power],
        'min_power': [min_all_data_power],
        'max_doppler': [max_all_data_doppler],
        'min_doppler': [min_all_data_doppler]}
df_max_min = pd.DataFrame(data)
df_max_min.to_csv('max_min.csv', index=False)
print("max power: {}".format(max_all_data_power))
print("min power: {}".format(min_all_data_power))
print("max doppler: {}".format(max_all_data_doppler))
print("min doppler: {}".format(min_all_data_doppler))
