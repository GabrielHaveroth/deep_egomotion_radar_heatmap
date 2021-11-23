from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from data_loader import *
from dataset_loaders import *
import math


def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.75
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    print('Learning rate: ' + str(lrate))
    return lrate


def create_heatmap_feature_extrator(input):
    net = layers.Conv2D(16, (3, 3), activation='relu')(input)
    net = layers.MaxPooling2D((2, 2))(net)
    net = layers.Conv2D(32, (3, 3), activation='relu')(net)
    net = layers.MaxPooling2D((2, 2))(net)
    net = layers.Conv2D(64, (3, 3), activation='relu')(net)
    net = layers.Flatten()(net)
    return net


# Loading dataset
seqs = ['2_22_2021_longboard_run7/', '2_23_2021_edgar_classroom_run5/']
base_path = '/home/lactec/dados/mestrado_gabriel/'
all_radar_params = get_cascade_params('/data/Conjuntos_Dados_Mestrado/calib')
radar_heatmap_params = all_radar_params['heatmap']
MIN_TIME_BETWEEN_PAIR = 0.4
MAX_TIME_BETWEEN_PAIR = 1
time_window = 0.4
init_time_stamp = 0
unique_pair = True
data = {}
pairs = []
files = []
for seq in seqs:
    name = path + seq
    gt_params = get_groundtruth_params()
    radar_timestamps = get_timestamps(name, radar_heatmap_params)
    gt_timestamps = get_timestamps(name, gt_params)
    # get groundtruth poses
    gt_poses = get_groundtruth(name)
    # interpolate groundtruth poses for each sensor measurement
    radar_gt, radar_indices = interpolate_poses(gt_poses,
                                                gt_timestamps,
                                                radar_timestamps)
    selected_timestamps = [radar_timestamps[index] for index in radar_indices]
    for i in range(len(selected_timestamps)):
        for j in range(i + 1, len(selected_timestamps)):
            timestamp_diff = round(
                selected_timestamps[j] - selected_timestamps[i], 2)
            if MIN_TIME_BETWEEN_PAIR <= timestamp_diff <= MAX_TIME_BETWEEN_PAIR:
                pairs.append((i, j))
                files.append(name)
                if unique_pair:
                    break

data['pair'] = pairs
data['file'] = files
df = pd.DataFrame(data)
traingen = RadarEgomotionDataGenerator(df, all_radar_params, batch_size=32)

# Create a learning rate schedule
lrate = LearningRateScheduler(step_decay)
checkpointer = ModelCheckpoint(filepath=path, monitor='val_loss',
                               mode='min', save_best_only=True, verbose=1)
# Cartesian heatmap
input_shape1 = (128, 128, 4)
# Range-elevation heatmap
input_shape2 = (32, 128, 4)
input1 = Input(input_shape1)
input2 = Input(input_shape2)
# CNN's
cnn1 = create_heatmap_feature_extrator(input1)
cnn2 = create_heatmap_feature_extrator(input2)
combined_cnn = layers.concatenate([cnn1, cnn2])
# Translation regressor
fc_trans = layers.Dense(128, activation='relu')(combined_cnn)  # tanh
fc_trans = layers.Dropout(0.25)(fc_trans)
fc_trans = layers.Dense(64, activation='relu')(fc_trans)
fc_trans = layers.Dense(3, activation='relu', name='fc_trans')(fc_trans)
# Rotation regressor
fc_rot = layers.Dense(128, activation='relu')(combined_cnn)  # tanh
fc_rot = layers.Dropout(0.25)(fc_rot)
fc_rot = layers.Dense(64, activation='relu')(fc_rot)
fc_rot = layers.Dense(3, activation='relu', name='fc_rot')(fc_rot)
model = Model(inputs=[input1, input2], outputs=[fc_trans, fc_rot])
model.summary()
rms_prop = RMSprop(learning_rate=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rms_prop, loss={'fc_trans': 'mse', 'fc_rot': 'mse'})
model.fit(traingen, epochs=100, callbacks=lrate)
