import tensorflow as tf
from helpers import * 
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1

def decode_heatmap(hm_bytes):
    frame_vals = tf.io.decode_raw(hm_bytes, little_endian=True, out_type=tf.float32)
    frame = tf.reshape(frame_vals, (32, 128, 128, 2))[:, :, :, 0]
    return frame


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# Create a dictionary with features that may be relevant.


def define_features(hm1_string, hm2_string, delta_pose):
    feature = {'heatmap1': _bytes_feature(hm1_string),
               'heatmap2': _bytes_feature(hm2_string),
               'delta_roll': _float_feature(delta_pose[0]),
               'delta_pitch': _float_feature(delta_pose[1]),
               'delta_yaw': _float_feature(delta_pose[2]),
               'delta_x': _float_feature(delta_pose[3]),
               'delta_y': _float_feature(delta_pose[4]),
               'delta_z': _float_feature(delta_pose[5]),
               'imu_idx1': _int64_feature(delta_pose[5]),
               'imu_idx2': _int64_feature()}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_heatmap_tfrecord(filename, items):
    with tf.io.TFRecordWriter(filename) as writer:
        for item in items:
            filename1 = item[1] + '/cascade/heatmaps/data/heatmap_' + str(item[0][0]) + '.bin'
            filename2 = item[1] + '/cascade/heatmaps/data/heatmap_' + str(item[0][1]) + '.bin'
            heatmap1_string = tf.io.read_file(filename1)
            heatmap2_string = tf.io.read_file(filename2)
            tf_example = define_features(heatmap1_string, heatmap2_string, item[4])
            writer.write(tf_example.SerializeToString())

def read_tfrecord(example):
    feature_description = {'heatmap1': tf.io.FixedLenFeature([], tf.string),
                           'heatmap2': tf.io.FixedLenFeature([], tf.string),
                           'delta_roll': tf.io.FixedLenFeature([], tf.float32),
                           'delta_pitch': tf.io.FixedLenFeature([], tf.float32),
                           'delta_yaw': tf.io.FixedLenFeature([], tf.float32),
                           'delta_x': tf.io.FixedLenFeature([], tf.float32),
                           'delta_y': tf.io.FixedLenFeature([], tf.float32),
                           'delta_z': tf.io.FixedLenFeature([], tf.float32)}
    example = tf.io.parse_single_example(example, feature_description)
    hm1 = (tf.reshape(decode_heatmap(example["heatmap1"]), ((32, 128, 128, 1))) - MIN_POWER) / (MAX_POWER - MIN_POWER)
    hm2 = (tf.reshape(decode_heatmap(example["heatmap2"]), ((32, 128, 128, 1))) - MIN_POWER) / (MAX_POWER - MIN_POWER)
    hm12 = tf.concat([hm1, hm2], -1)
    y_rot = tf.stack([example['delta_roll'], example['delta_pitch'], example['delta_yaw']])
    y_trans = tf.stack([example['delta_x'], example['delta_y'], example['delta_z']])
    return hm12, (y_trans, y_rot)


def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
    # dataset = dataset.shuffle(128)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)

    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

def get_dataset(filenames):
    dataset = load_dataset(filenames)
    # dataset = dataset.cache()
    # dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset