

from tensorflow.keras import layers, Input, activations, Model
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

def build_sparse_features_regressor(input: Input) -> layers.Flatten:
    conv_1 = layers.Conv3D(name='conv1', filters=32, kernel_size=3, strides=1,
                           padding='same', activation=activations.relu)(input)
    conv_2 = layers.Conv3D(name='conv2', filters=32, kernel_size=3, strides=2,
                           padding='same', activation=activations.relu)(conv_1)
    pool_1 = layers.MaxPool3D(name="pool1", strides=2)(conv_2)
    conv_3 = layers.Conv3D(name='conv3', filters=64, kernel_size=3, strides=2,
                           padding='same', activation=activations.relu)(pool_1)
    conv_3_1 = layers.Conv3D(name='conv3_1', filters=64, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_3)
    pool_2 = layers.MaxPool3D(name="pool2", strides=2)(conv_3_1)
    conv_4 = layers.Conv3D(name='conv4', filters=128, kernel_size=3, strides=2,
                           padding='same', activation=activations.relu)(pool_2)
    conv_4_1 = layers.Conv3D(name='conv4_1', filters=128, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_4)

    heatmap_features = layers.Flatten()(conv_4_1)

       # Translation regressor
    fc_trans = layers.Dense(2048, activation='relu')(heatmap_features)  # tanh
    fc_trans = layers.Dropout(0.50)(fc_trans)
    fc_trans = layers.Dense(2048, activation='relu')(fc_trans)
    fc_trans = layers.Dense(3, activation='linear', name='fc_trans')(fc_trans)
    # Rotation regressor
    fc_rot = layers.Dense(2048, activation='relu')(heatmap_features)  # tanh
    fc_rot = layers.Dropout(0.50)(fc_rot)
    fc_rot = layers.Dense(2048, activation='relu')(fc_rot)
    fc_rot = layers.Dense(3, activation='linear', name='fc_rot')(fc_rot)
    model = Model(inputs=input, outputs=[fc_trans, fc_rot])
    return model


def build_6D_pose_regressor(sensor_features: layers.Flatten()) -> Tuple[layers.Dense, layers.Dense]:
    # Translation regressor
    fc_trans = layers.Dense(2048, activation='relu')(sensor_features)  # tanh
    fc_trans = layers.Dropout(0.50)(fc_trans)
    fc_trans = layers.Dense(1024, activation='relu')(fc_trans)
    fc_trans = layers.Dense(3, activation='linear', name='fc_trans')(fc_trans)
    # Rotation regressor
    fc_rot = layers.Dense(2048, activation='relu')(sensor_features)  # tanh
    fc_rot = layers.Dropout(0.50)(fc_rot)
    fc_rot = layers.Dense(1024, activation='relu')(fc_rot)
    fc_rot = layers.Dense(3, activation='linear', name='fc_rot')(fc_rot)
    return fc_trans, fc_rot

def build_2D_pose_regressor(sensor_features: layers.Flatten()) -> Tuple[layers.Dense, layers.Dense]:
    # Translation regressorfc_rottivation='relu')(fc_rot)
    fc_rot = layers.Dense(1, activation='linear', name='fc_rot')(fc_rot)
    return fc_trans, fc_rot

def build_2D_flownet(input: Input, flatten=True):
    conv_1 = layers.Conv2D(name='conv1', filters=64, kernel_size=7, strides=2,
                           padding='same', activation=activations.relu)(input)
    conv_2 = layers.Conv2D(name='conv2', filters=128, kernel_size=5, strides=2,
                           padding='same', activation=activations.relu)(conv_1)
    conv_3 = layers.Conv2D(name='conv3', filters=256, kernel_size=5, strides=2,
                           padding='same', activation=activations.relu)(conv_2)
    conv_3_1 = layers.Conv2D(name='conv3_1', filters=256, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_3)
    conv_4 = layers.Conv2D(name='conv4', filters=512, kernel_size=3, strides=2,
                           padding='same', activation=activations.relu)(conv_3_1)
    conv_4_1 = layers.Conv2D(name='conv4_1', filters=512, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_4)
    conv_5 = layers.Conv2D(name='conv5', filters=512, kernel_size=3, strides=2,
                           padding='same', activation=activations.relu)(conv_4_1)
    conv_5_1 = layers.Conv2D(name='conv5_1', filters=512, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_5)
    conv_6 = layers.Conv2D(name='conv6', filters=1024, kernel_size=3, strides=2,
                           padding='same', activation=activations.relu)(conv_5_1)
    conv_6_1 = layers.Conv2D(name='conv6_1', filters=1024, kernel_size=3,
                             strides=1, padding='same', activation=activations.relu)(conv_6)
    if flatten:
        flow2D_features = layers.Flatten()(conv_6_1)
    else:
        flow2D_features = conv_6_1

    return flow2D_features


def feature_extractor_2D_heatmap(input: Input):
    x = layers.Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
    x = layers.Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    x = layers.Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    x = layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    x = layers.Flatten()(x) 
    fc_trans = layers.Dense(2048, activation='relu')(x)  # tanh
    fc_trans = layers.Dropout(0.25)(fc_trans)
    fc_trans = layers.Dense(1024, actinputation='relu')(x)  # tanh
    fc_rot = layers.Dropout(0.25)(fc_rot)
    fc_rot = layers.Dense(64, activation='relu')(fc_rot)
    fc_rot = layers.Dense(1, activation='linear', name='fc_rot')(fc_rot)
    
    model = Model(inputs=input, outputs=[fc_trans, fc_rot])

    return model

# Build CROSS-attentive multi-modal odom with IMU always used
def build_model_cross_att(imu_length, input_heatmap, mask_att='sigmoid', istraining=True, write_mask=False):
    # --- panoramic image data
    net = build_2D_flownet(input_heatmap, flatten=False)

    # generate the mask for visual features
    visual_mask = layers.GlobalAveragePooling2D()(net) # reshape to (?, 1, 1024), 1 stands for timeDistr.
    visual_mask = layers.Dense(int(1024/256), activation='relu', use_bias=False, name='visual_mask_relu')(visual_mask)
    visual_mask = layers.Dense(1024, activation='sigmoid', use_bias=False, name='visual_mask_sigmoid')(visual_mask)
    visual_mask = layers.Reshape((1, 1, 1, 1024))(visual_mask)

    # activate mask by element-wise multiplication
    visual_att_fea = layers.Multiply()([net, visual_mask])
    visual_att_fea = layers.Flatten()(visual_att_fea)

    # IMU data
    imu_data = Input(shape=(imu_length, 6), name='imu_data')
    imu_lstm_1 = layers.LSTM(128, return_sequences=True, name='imu_lstm_1')(imu_data)  # 128, 256

    # channel-wise IMU attention
    reshape_imu = layers.Reshape((1, imu_length * 128))(imu_lstm_1)  # 2560, 5120, 10240
    imu_mask = layers.Dense(128, activation='relu', use_bias=False, name='imu_mask_relu')(reshape_imu)
    imu_mask = layers.Dense(imu_length * 128, activation='sigmoid', use_bias=False, name='imu_mask_sigmoid')(imu_mask)
    imu_att_fea = layers.Multiply()([reshape_imu, imu_mask])

    # cross-modal attention
    imu4visual_mask = layers.Dense(128, activation='relu', use_bias=False, name='imu4visual_mask_relu')(imu_att_fea)
    imu4visual_mask = layers.Dense(4096, activation=mask_att, use_bias=False, name='imu4visual_mask_sigmoid')(imu4visual_mask)
    cross_visual_fea = layers.Multiply()([visual_att_fea, imu4visual_mask])

    visual4imu_mask = layers.Dense(128, activation='relu', use_bias=False, name='visual4imu_mask_relu')(visual_att_fea)
    visual4imu_mask = layers.Dense(imu_length * 128, activation=mask_att, use_bias=False, name='visual4imu_mask_sigmoid')(visual4imu_mask)
    cross_imu_fea = layers.Multiply()([imu_att_fea, visual4imu_mask])

    # Standard merge feature
    merge_features = layers.concatenate([cross_visual_fea, cross_imu_fea], mode='concat', concat_axis=-1, name='merge_features')

    # Selective features
    forward_lstm_1 = layers.LSTM(512, dropout_W=0.25, return_sequences=True, name='forward_lstm_1')(
    merge_features = layers.concatenate([cross_visual_fea, cross_imu_fea], axis=-1)

    # Selective features
    forward_lstm_1 = layers.LSTM(512, dropout=0.25, return_sequences=True, name='forward_lstm_1')(
        merge_features)  # dropout_W=0.2, dropout_U=0.2
    forward_lstm_2 = layers.LSTM(512, return_sequences=True, name='forward_lstm_2')(forward_lstm_1)

    fc_position_1 = layers.Dense(128, activation='relu', name='fc_position_1')(forward_lstm_2)  # tanh
    dropout_pos_1 = layers.Dropout(0.25, name='dropout_pos_1')(fc_position_1)
    fc_position_2 = layers.Dense(64, activation='relu', name='fc_position_2')(dropout_pos_1)  # tanh
    fc_trans = layers.Dense(3, name='fc_trans')(fc_position_2)

    fc_orientation_1 = layers.Dense(128, activation='relu', name='fc_orientation_1')(forward_lstm_2)  # tanh
    dropout_orientation_1 = layers.Dropout(0.25, name='dropout_wpqr_1')(fc_orientation_1)
    fc_orientation_2 = layers.Dense(64, activation='relu', name='fc_orientation_2')(dropout_orientation_1)  # tanh
    fc_rot = layers.Dense(3, name='fc_rot')(fc_orientation_2)

    model = Model(inputs=[input_heatmap, imu_data], outputs=[fc_trans, fc_rot])

    return model