from scipy.spatial.transform import Rotation, Slerp
from geometric_operations import *
from dataset_loaders import *
import scipy
import scipy.ndimage
import multiprocessing as mp
import numpy as np
import pandas as pd

MAX_POWER = 4268728.0
MIN_POWER = 3.081555128097534
MAX_DOPPLER = 1.1527502536773682
MIN_DOPPLER = -1.317428708076477


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
        c = ((tgt_stamps[tgt_idx] - src_stamps[src_idx]) /
             (src_stamps[src_idx+1] - src_stamps[src_idx]))

        # interpolate position
        pose = np.eye(4)
        pose[:3, 3] = ((1.0 - c) * src_poses[src_idx]
                       ['position'] + c * src_poses[src_idx+1]['position'])

        # interpolate orientation
        r_src = Rotation.from_quat([src_poses[src_idx]['orientation'],
                                    src_poses[src_idx+1]['orientation']])
        slerp = Slerp([0, 1], r_src)
        pose[:3, :3] = slerp([c])[0].as_dcm()

        tgt_poses.append(pose)

        # advance target index
        tgt_idx += 1

    tgt_indices = range(tgt_start_idx, tgt_end_idx + 1)
    return tgt_poses, tgt_indices


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
    return radar_heatmap[16, :, :, :]


def get_cartesian_slice_heatmap(radar_heatmap):
    radar_heatmap_slice_power = radar_heatmap[16, :, :, 0]
    radar_heatmap_slice_doppler = radar_heatmap[16, :, :, 1]
    cartesian_heatmap_power = scipy.ndimage.geometric_transform(radar_heatmap_slice_power, polar2cartesian, order=3,
                                                                output_shape=(radar_heatmap_slice_power.shape[0],
                                                                              radar_heatmap_slice_power.shape[1]),
                                                                extra_keywords={'inputshape': radar_heatmap_slice_power.shape,
                                                                                'origin': (radar_heatmap_slice_power.shape[0],
                                                                                           radar_heatmap_slice_power.shape[1] / 2)})
    cartesian_heatmap_doppler = scipy.ndimage.geometric_transform(radar_heatmap_slice_doppler, polar2cartesian, order=3,
                                                                  output_shape=(radar_heatmap_slice_doppler.shape[0],
                                                                                radar_heatmap_slice_doppler.shape[1]),
                                                                  extra_keywords={'inputshape': radar_heatmap_slice_doppler.shape,
                                                                                  'origin': (radar_heatmap_slice_doppler.shape[0],
                                                                                             radar_heatmap_slice_doppler.shape[1] / 2)})

    cartesian_heatmap_power = np.reshape(
        cartesian_heatmap_power, (128, 128, 1))
    cartesian_heatmap_doppler = np.reshape(
        cartesian_heatmap_doppler, (128, 128, 1))
    cartesian_heatmap = np.concatenate(
        (cartesian_heatmap_power, cartesian_heatmap_doppler), axis=-1)
    return cartesian_heatmap


def get_cartesian_heatmap(radar_heatmap):
    max_heatmap = radar_heatmap.mean(axis=0)
    cartesian_hm = scipy.ndimage.geometric_transform(max_heatmap, polar2cartesian, order=3,
                                                     output_shape=(max_heatmap.shape[0],
                                                                   max_heatmap.shape[1]),
                                                     extra_keywords={'inputshape': max_heatmap.shape,
                                                                     'origin': (max_heatmap.shape[0], max_heatmap.shape[1] / 2)})
    return cartesian_hm

def get_range_azimute_slice_heatmap(radar_heatmap):
    return radar_heatmap[16, :, :, :]


def get_range_elevation_slice_heatmap(radar_heatmap):
    return radar_heatmap[:, 64, :, :]


def get_heatmap_poses(name, ht_params):
    gt_poses = get_groundtruth(name)
    radar_timestamps = get_timestamps(name, ht_params)
    gt_timestamps = get_timestamps(name, ht_params)
    radar_gt, _ = interpolate_poses(gt_poses,
                                    gt_timestamps,
                                    radar_timestamps)

    return radar_gt


def get_data_3D_heatmap_gt(row, ht_params, poses_seq):
    seq = row['file']
    pair = row['heatmap_pairs']
    poses = poses_seq[seq]
    power_hm_t1 = get_heatmap(pair[0], seq, ht_params)
    power_hm_t1 = (power_hm_t1[:, :, :, 0].reshape(
        (32, 128, 128, 1)) - MIN_POWER) / (MAX_POWER - MIN_POWER)
    radar_hm_t2 = get_heatmap(pair[1], seq, ht_params)
    power_hm_t2 = (radar_hm_t2[:, :, :, 0].reshape(
        (32, 128, 128, 1)) - MIN_POWER) / (MAX_POWER - MIN_POWER)
    power_hm_t12 = np.concatenate([power_hm_t1,
                                   power_hm_t2], -1)
    delta_pose, _ = get_ground_6d_poses_euler(poses[pair[0]],
                                              poses[pair[1]])
    # delta_rot.append(delta_pose[0:3])
    # delta_trans.append(delta_pose[3:])
    return power_hm_t12, delta_pose


def get_data_3D_batch_gt(batches, hm_params, poses_seq):
    delta_poses = []
    hm_powers_t12 = []
    with mp.Pool(mp.cpu_count()) as pool:
        results_async = [pool.apply_async(get_data_3D_heatmap_gt, args=(
            row, hm_params, poses_seq, )) for _, row in batches.iterrows()]
        for result in results_async:
            hm_power_t12, delta_pose = result.get()
            delta_poses.append(delta_pose)
            hm_powers_t12.append(hm_power_t12)
    return delta_poses, hm_powers_t12


def get_data_2D_cart_heatmap_gt(row, ht_params, poses_seq):
    seq = row['file']
    pair = row['heatmap_pairs']
    poses = poses_seq[seq]
    power_hm_t1 = (get_heatmap(pair[0], seq, ht_params)[:, :, :, 0] - MIN_POWER) / (MAX_POWER - MIN_POWER)
    cart_hm_t1 = get_cartesian_heatmap(power_hm_t1).reshape((128, 128, 1))
    power_hm_t2 = (get_heatmap(pair[1], seq, ht_params)[:, :, :, 0] - MIN_POWER) / (MAX_POWER - MIN_POWER)
    cart_hm_t2 = get_cartesian_heatmap(power_hm_t2).reshape((128, 128, 1))
    cart_hm_t12 = np.concatenate([cart_hm_t1,
                                  cart_hm_t2], -1)
    delta_pose, _ = get_ground_6d_poses_euler(poses[pair[0]],
                                              poses[pair[1]])
    # delta_rot.append(delta_pose[0:3])
    # delta_trans.append(delta_pose[3:])]
    return cart_hm_t12, delta_pose
