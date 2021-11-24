from numpy.lib.function_base import angle
from tensorflow.keras.optimizers import RMSprop
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


def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.75
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    print('Learning rate: ' + str(lrate))
    return lrate


# Loading dataset
all_radar_params = get_cascade_params('/data/Conjuntos_Dados_Mestrado/calib')
radar_heatmap_params = all_radar_params['heatmap']
df_data = pd.read_pickle('./models/train.pkl')
train_df, val_df = train_test_split(df_data, test_size=0.15, random_state=42)
train_gen = RadarEgomotionDataGenerator(df_data,
                                        all_radar_params,
                                        batch_size=8,
                                        data_type='2d_cart_heatmap')

val_gen = RadarEgomotionDataGenerator(val_df,
                                      all_radar_params,
                                      batch_size=8,
                                      data_type='2d_cart_heatmap')

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
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# Create a learning rate schedule
lrate = LearningRateScheduler(step_decay)
input_shape = (128, 128, 2)
input = Input(input_shape)
flownet3D_features = nets.build_2D_flownet(input)
fc_trans, fc_rot = nets.build_2D_pose_regressor(flownet3D_features)
model = Model(inputs=input, outputs=[fc_trans, fc_rot])
model.summary()
rms_prop = RMSprop(learning_rate=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rms_prop, loss={'fc_trans': 'mse', 'fc_rot': 'mse'})
# history_model = model.fit(train_gen, validation_data=val_gen, epochs=2, callbacks=[lrate, checkpointer, tensorboard_callback], verbose=1)
history_model = model.fit(train_gen, validation_data=val_gen, epochs=100)
history_file = open("history.pkl", "wb")
pickle.dump(history_model, history_file)
