

from tensorflow.keras import layers, Input, activations
from typing import Tuple


def build_3D_flownet(input: Input) -> layers.Flatten:
    conv_1 = layers.Conv3D(name='conv1', filters=64, kernel_size=7, strides=2,
                           padding='same', activation=activations.relu)(input)
    conv_2 = layers.Conv3D(name='conv2', filters=128, kernel_size=5, strides=2,
                           padding='same', activation=activations.relu)(conv_1)
    conv_3 = layers.Conv3D(name='conv3', filters=256, kernel_size=5, strides=2,
                           padding='same', activation=activations.relu)(conv_2)
    conv_3_1 = layers.Conv3D(name='conv3_1', filters=256, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_3)
    conv_4 = layers.Conv3D(name='conv4', filters=512, kernel_size=3, strides=2,
                           padding='same', activation=activations.relu)(conv_3_1)
    conv_4_1 = layers.Conv3D(name='conv4_1', filters=512, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_4)
    conv_5 = layers.Conv3D(name='conv5', filters=512, kernel_size=3, strides=2,
                           padding='same', activation=activations.relu)(conv_4_1)
    conv_5_1 = layers.Conv3D(name='conv5_1', filters=512, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_5)
    conv_6 = layers.Conv3D(name='conv6', filters=1024, kernel_size=3, strides=2,
                           padding='same', activation=activations.relu)(conv_5_1)
    conv_6_1 = layers.Conv3D(name='conv6_1', filters=1024, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_6)
    flow3D_features = layers.Flatten()(conv_6_1)
    return flow3D_features


def build_6D_pose_regressor(sensor_features: layers.Flatten()) -> Tuple[layers.Dense, layers.Dense]:
    # Translation regressor
    fc_trans = layers.Dense(128, activation='relu')(sensor_features)  # tanh
    fc_trans = layers.Dropout(0.25)(fc_trans)
    fc_trans = layers.Dense(64, activation='relu')(fc_trans)
    fc_trans = layers.Dense(3, activation='linear', name='fc_trans')(fc_trans)
    # Rotation regressor
    fc_rot = layers.Dense(128, activation='relu')(sensor_features)  # tanh
    fc_rot = layers.Dropout(0.25)(fc_rot)
    fc_rot = layers.Dense(64, activation='relu')(fc_rot)
    fc_rot = layers.Dense(3, activation='linear', name='fc_rot')(fc_rot)
    return fc_trans, fc_rot
