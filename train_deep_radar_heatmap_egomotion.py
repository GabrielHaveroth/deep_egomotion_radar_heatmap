from operator import ge
from dataset_loaders import *
import math
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation, Slerp
import scipy
import scipy.ndimage
from geometric_operations import *
# calculates point locations in the sensor frame for plotting heatmaps
# param[in] params: heatmap parameters for the sensor
# return pcl: the heatmap point locations
def get_heatmap_points(params):

  # transform range-azimuth-elevation heatmap to pointcloud
  pcl = np.zeros([params['num_elevation_bins'],
                  params['num_azimuth_bins'],
                  params['num_range_bins'] - args.min_range,
                  5])

  for range_idx in range(params['num_range_bins'] - args.min_range):
    for az_idx in range(params['num_azimuth_bins']):
      for el_idx in range(params['num_elevation_bins']):
        pcl[el_idx,az_idx,range_idx,:3] = polar_to_cartesian(range_idx + args.min_range, az_idx, el_idx, params)

  pcl = pcl.reshape(-1,5)
  return pcl
  
def interpolate_poses(src_poses, src_stamps, tgt_stamps):
  src_start_idx = 0
  tgt_start_idx = 0
  src_end_idx = len(src_stamps) - 1
  tgt_end_idx = len(tgt_stamps) - 1

  # ensure first source timestamp is immediately before first target timestamp
  while tgt_start_idx < tgt_end_idx and tgt_stamps[tgt_start_idx] < src_stamps[src_start_idx]:
    tgt_start_idx += 1

  # ensure last source timestamp is immediately after last target timestamp
  while tgt_end_idx > tgt_start_idx and tgt_stamps[tgt_end_idx] > src_stamps[src_end_idx]:
    tgt_end_idx -= 1

  # iterate through target timestamps, 
  # interpolating a pose for each as a 4x4 transformation matrix
  tgt_idx = tgt_start_idx
  src_idx = src_start_idx
  tgt_poses = []
  while tgt_idx <= tgt_end_idx and src_idx <= src_end_idx:
    # find source timestamps bracketing target timestamp
    while src_idx + 1 <= src_end_idx and src_stamps[src_idx + 1] < tgt_stamps[tgt_idx]:
      src_idx += 1

    # get interpolation coefficient
    c = ((tgt_stamps[tgt_idx] - src_stamps[src_idx]) 
          / (src_stamps[src_idx+1] - src_stamps[src_idx]))

    # interpolate position
    pose = np.eye(4)
    pose[:3,3] = ((1.0 - c) * src_poses[src_idx]['position'] 
                        + c * src_poses[src_idx+1]['position'])

    # interpolate orientation

    r_src = Rotation.from_quat([src_poses[src_idx]['orientation'],
                            src_poses[src_idx+1]['orientation']])
    slerp = Slerp([0,1],r_src)
    pose[:3,:3] = slerp([c])[0].as_dcm()

    tgt_poses.append(pose)

    # advance target index
    tgt_idx += 1

  tgt_indices = range(tgt_start_idx, tgt_end_idx + 1)
  return tgt_poses, tgt_indices

def polar_to_cartesian(r_bin, az_bin, el_bin, params):
  point = np.zeros(3)
  point[0] = (r_bin * params['range_bin_width'] 
              * math.cos(params['elevation_bins'][el_bin]) 
              * math.cos(params['azimuth_bins'][az_bin]))
  point[1] = (r_bin * params['range_bin_width']
              * math.cos(params['elevation_bins'][el_bin])
              * math.sin(params['azimuth_bins'][az_bin]))
  point[2] = (r_bin * params['range_bin_width']
              * math.sin(params['elevation_bins'][el_bin]))
  return point

def polar2cartesian(outcoords, inputshape, origin):
    """Coordinate transform for converting a polar array to Cartesian coordinates. 
    inputshape is a tuple containing the shape of the polar array. origin is a
    tuple containing the x and y indices of where the origin should be in the
    output array."""

    xindex, yindex = outcoords
    x0, y0 = origin
    x = xindex - x0
    y = yindex - y0

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_index = np.round((theta + np.pi) * inputshape[1] / (2 * np.pi))

    return (r, theta_index)

def get_range_doppler_slice_heatmap(radar_heatmap):
  return radar_heatmap[16, 64, :, :]

def get_cartesian_slice_heatmap(radar_heatmap):
  radar_heatmap_slice = radar_heatmap[16, :, :, 0]
  cartesian_heatmap = scipy.ndimage.geometric_transform(radar_heatmap_slice, polar2cartesian, order=3, output_shape = (radar_heatmap_slice.shape[0], radar_heatmap_slice.shape[0]), extra_keywords = {'inputshape':radar_heatmap_slice.shape,'origin':(radar_heatmap_slice.shape[0], radar_heatmap_slice.shape[1] / 2)})
  return cartesian_heatmap

def get_range_azimute_slice_heatmap(radar_heatmap):
  return radar_heatmap[16, :, :, 0]

def get_range_elevation_slice_heatmap(radar_heatmap):
  return radar_heatmap[:, 64, :, 0]

seq = '/mnt/Share_Data/Conjuntos_Dados_Mestrado/2_23_2021_edgar_classroom_run5/'
all_radar_params = get_cascade_params('/mnt/Share_Data/Conjuntos_Dados_Mestrado/calib')
radar_heatmap_params = all_radar_params['heatmap']
gt_params = get_groundtruth_params()
radar_timestamps = get_timestamps(seq, radar_heatmap_params)
gt_timestamps = get_timestamps(seq, gt_params)
# get groundtruth poses
gt_poses = get_groundtruth(seq)
# interpolate groundtruth poses for each sensor measurement
radar_gt, radar_indices = interpolate_poses(gt_poses, 
                                            gt_timestamps, 
                                            radar_timestamps)

MIN_TIME_BETWEEN_PAIR = 0.4
MAX_TIME_BETWEEN_PAIR = 1
selected_timestamps = [radar_timestamps[index] for index in radar_indices]
time_window = 0.4
init_time_stamp = 0
unique_pair = True
pairs = []

for i in range(len(selected_timestamps)):
      for j in range(i + 1, len(selected_timestamps)):
        timestamp_diff = round(selected_timestamps[j] - selected_timestamps[i], 2)
        if  MIN_TIME_BETWEEN_PAIR <= timestamp_diff <= MAX_TIME_BETWEEN_PAIR:
            pairs.append((int(i), int(j)))
            if unique_pair:
              break

# for pair in pairs:
#   delta_pose, T12 = get_ground_6d_poses_euler(radar_gt[pair[0]], radar_gt[pair[0]])
#   radar_heatmap_t_1 = get_heatmap(pair[0], seq, radar_heatmap_params)
#   radar_heatmap_t_2 = get_heatmap(pair[1], seq, radar_heatmap_params)
#   cart_radar_heatmap_t_1 = get_cartesian_slice_heatmap(radar_heatmap_t_1)
#   cart_radar_heatmap_t_2 = get_cartesian_slice_heatmap(radar_heatmap_t_2)
#   elev_radar_heatmap_t_1  = get_range_elevation_slice_heatmap(radar_heatmap_t_1)
#   elev_radar_heatmap_t_2  = get_range_elevation_slice_heatmap(radar_heatmap_t_2)
#   vel_radar_heatmap_t_1 = get_range_doppler_slice_heatmap(radar_heatmap_t_1)
#   vel_radar_heatmap_t_2 = get_range_doppler_slice_heatmap(radar_heatmap_t_2)


# ranges = np.array([i*radar_heatmap_params['range_bin_width'] for i in range(radar_heatmap_params['num_range_bins'])])
# azimuths = radar_heatmap_params['azimuth_bins']
# elevation = radar_heatmap_params['elevation_bins']
# min_index = elevation.index(np.min(np.abs(elevation)))
# radar_heatmap = get_heatmap(0, seq, radar_heatmap_params)
# selected_timestamps = [radar_timestamps[index] for index in radar_indices]
# cartesian_heatmap = get_cartesian_slice_heatmap(radar_heatmap)


# # Animantion
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig = plt.figure()
# # im = plt.imshow(cartesian_heatmap, animated=True)
# im = plt.imshow(radar_heatmap[16, :, :, 0], animated=True)
# actual_index = 0

# def updatefig(*args):
#     global radar_indices
#     global actual_index
#     global radar_heatmap_params
#     global seq
#     actual_index = actual_index + 1
#     radar_heatmap = get_heatmap(radar_indices[actual_index], seq, radar_heatmap_params)
#     # _heatmap = get_cartesian_slice_heatmap(radar_heatmap)
#     _heatmap = radar_heatmap[16, :, :, 0]
#     if actual_index >= len(radar_indices):
#       actual_index = 0
#     im.set_array(_heatmap)
#     return im,

# ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
# plt.show()