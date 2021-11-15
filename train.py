from tensorflow.keras import layers, Input, Model, activations
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.regularizers import get
from sklearn.model_selection import train_test_split
from data_loader import *
from dataset_loaders import *
import datetime
import math
import pickle
import os


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


def create_3D_flownet(input):
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
    flow_net = layers.Flatten()(conv_6_1)
    return flow_net

# Loading dataset
seqs = ['2_22_2021_longboard_run7/', '2_23_2021_edgar_classroom_run5/']
path = '/data/Conjuntos_Dados_Mestrado/'
all_radar_params = get_cascade_params('/data/Conjuntos_Dados_Mestrado/calib')
radar_heatmap_params = all_radar_params['heatmap']
df_data = pd.read_pickle('train.pkl')
train_df, val_df = train_test_split(df_data, test_size=0.15, random_state=42)
train_gen = RadarEgomotionDataGenerator(df_data,
                                        all_radar_params,
                                        batch_size=16,
                                        data_type='3d_heatmap')

val_gen = RadarEgomotionDataGenerator(val_df,
                                       all_radar_params,
                                       batch_size=16,
                                       data_type='3d_heatmap')

# Creating checkpoint to save the best model
MODELS_FOLDER = './models'
checkpoint_folder = os.path.sep.join([MODELS_FOLDER, "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
if os.path.exists(checkpoint_folder):
    os.remove(checkpoint_folder)
checkpointer = ModelCheckpoint(filepath=checkpoint_folder, monitor='val_loss',
                                   mode='min', save_best_only=True, verbose=1)
# Creating tensorboard to realtime evaluate the model
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# Create a learning rate schedule
lrate = LearningRateScheduler(step_decay)
# # Cartesian heatmap
# input_shape1 = (128, 128, 4)
# # Range-elevation heatmap
# input_shape2 = (32, 128, 4)
# input1 = Input(input_shape1)
# input2 = Input(input_shape2)
# # CNN's
# cnn1 = create_heatmap_feature_extrator(input1)
# cnn2 = create_heatmap_feature_extrator(input2)
# combined_cnn = layers.concatenate([cnn1, cnn2])
input_shape = (32, 128, 128, 2)
input = Input(input_shape)
flow_net = create_3D_flownet(input)
# Translation regressor
fc_trans = layers.Dense(128, activation='relu')(flow_net)  # tanh
fc_trans = layers.Dropout(0.25)(fc_trans)
fc_trans = layers.Dense(64, activation='relu')(fc_trans)
fc_trans = layers.Dense(3, activation='relu', name='fc_trans')(fc_trans)
# Rotation regressor
fc_rot = layers.Dense(128, activation='relu')(flow_net)  # tanh
fc_rot = layers.Dropout(0.25)(fc_rot)
fc_rot = layers.Dense(64, activation='relu')(fc_rot)
fc_rot = layers.Dense(3, activation='relu', name='fc_rot')(fc_rot)
model = Model(inputs=input, outputs=[fc_trans, fc_rot])
model.summary()
rms_prop = RMSprop(learning_rate=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rms_prop, loss={'fc_trans': 'mse', 'fc_rot': 'mse'})
# history_model = model.fit(train_gen, validation_data=val_gen, epochs=2, callbacks=[lrate, checkpointer, tensorboard_callback], verbose=1)
history_model = model.fit(train_gen, validation_data=val_gen, epochs=2)
history_file = open("history.pkl", "wb")
pickle.dump(history_model, history_file)