from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras import Model, Input
from sklearn.model_selection import train_test_split
from data_loader import *
from dataset_loaders import *
import datetime
import math
import pickle
import os
import networks as nets

old_model = tf.keras.models.load_model('./models/weights-058-0.0320.hdf5')
weights = old_model.get_weights()


def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.75
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    print('Learning rate: ' + str(lrate))
    return lrate


# Loading dataset
all_radar_params = get_cascade_params('/home/lactec/dados/mestrado_gabriel/calib')
radar_heatmap_params = all_radar_params['heatmap']
df_data = pd.read_pickle('./metadata/train.pkl')
delta_poses = df_data['delta_poses_6D'].values.copy()
heatmaps = np.load('/home/lactec/dados/mestrado_gabriel/heatmap.npy')
imu_data = np.load('/home/lactec/dados/mestrado_gabriel/imu.npy')
delta_poses_arr = np.array(delta_poses)
# scaler = pickle.load(open('scaler.pkl', 'rb'))
# delta_poses_arr = scaler.transform(delta_poses_arr)
y = delta_poses_arr

heatmaps_train, heatmaps_val, imu_data_train, imu_data_val, y_train, y_val = train_test_split(heatmaps, imu_data, y, test_size=0.1, random_state=42)
y_rot_val = y_val[:, 2]
y_rot_train = y_train[:, 2]
y_trans_val = y_val[:, 0:2]
y_trans_train = y_train[:, 0:2]
# all_radar_params = get_cascade_params('/home/lactec/dados/mestrado_gabriel/calib')
# radar_heatmap_params = all_radar_params['heatmap']
df_data = pd.read_pickle('./metadata/train.pkl')
delta_poses = df_data['delta_poses_6D'].values.copy()
y = []
for delta_pose in delta_poses:
    y.append(delta_pose)
y = np.array(y)

heatmaps = np.load('/data/heatmap.npy')
imu_data = np.load('/data/imu.npy')

# scaler = pickle.load(open('scaler.pkl', 'rb'))
# delta_poses_arr = scaler.transform(delta_poses_arr)

heatmaps_train, heatmaps_val, imu_data_train, imu_data_val, y_train, y_val = train_test_split(heatmaps, imu_data, y, test_size=0.1, random_state=42)
y_rot_val = y_val[:, 0:3]
y_rot_train = y_train[:, 0:3]
y_trans_val = y_val[:, 3:]
y_trans_train = y_train[:, 3:]
y_train = [y_trans_train, y_rot_train]
y_val = [y_trans_val, y_rot_val]
X_train = [heatmaps_train, imu_data_train]
X_val = [heatmaps_val, imu_data_val]

# train_df, val_df = train_test_split(df_data, test_size=0.10, random_state=42)

# train_gen = RadarEgomotionDataGenerator(df_data,
#                                         all_radar_params,
#                                         batch_size=32,
#                                         data_type='2d_cart_heatmap')

# val_gen = RadarEgomotionDataGenerator(val_df,
#                                       all_radar_params,
#                                       batch_size=32,
#                                       data_type='2d_cart_heatmap')

# Creating checkpoint to save the best model
MODELS_FOLDER = './models'
checkpoint_folder = os.path.sep.join(
    [MODELS_FOLDER, "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
if os.path.exists(checkpoint_folder):
    os.remove(checkpoint_folder)
checkpointer = ModelCheckpoint(filepath=checkpoint_folder, monitor='val_loss',
                               mode='min', save_best_only=True, verbose=1)
# Creating tensorboard to realtime evaluate the model
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
# Create a learning rate schedule
lrate = LearningRateScheduler(step_decay)
input_shape = (128, 128, 2)
input = Input(input_shape)
# flownet2D_features = nets.build_2D_flownet(input)
# fc_trans, fc_rot = nets.build_2D_pose_regressor(flownet2D_features)
model = nets.b
model.summary()
adam_opt = Adam(learning_rate=0.001)
model.compile(optimizer=adam_opt, loss={'fc_trans': 'mse', 'fc_rot': 'mse'})
# model.set_weights(weights)
history_model = model.fit(X_train, y_train, validation_data=[X_val, y_val], epochs=300, batch_size=8, callbacks=[lrate, checkpointer, tensorboard_callback], verbose=1)
history_file = open("history.pkl", "wb")
pickle.dump(history_model, history_file)
