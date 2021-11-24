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
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    print('Learning rate: ' + str(lrate))
    return lrate
    
# Loading dataset
# seqs = ['2_22_2021_longboard_run7/', '2_23_2021_edgar_classroom_run5/']
# path = '/data/Conjuntos_Dados_Mestrado/'
data_path = '/home/lactec/dados/mestrado_gabriel/coloradar/'
calib_path = '/home/lactec/dados/mestrado_gabriel/calib'
all_radar_params = get_cascade_params(calib_path)
radar_heatmap_params = all_radar_params['heatmap']
df_data = pd.read_pickle('/home/lactec/Codigos_Mestrado_GabrielH/deep_egomotion_radar_heatmap/metadata/train.pkl')
train_df, val_df = train_test_split(df_data, test_size=0.15, random_state=42)
train_gen = RadarEgomotionDataGenerator(df_data,
                                        all_radar_params,
                                        batch_size=32,
                                        data_type='3d_heatmap')

val_gen = RadarEgomotionDataGenerator(val_df,
                                       all_radar_params,
                                       batch_size=32,
                                       data_type='3d_heatmap')

# Creating checkpoint to save the best model
MODELS_FOLDER = '/home/lactec/dados/mestrado_gabriel/coloradar/models'
checkpoint_folder = os.path.sep.join([MODELS_FOLDER, "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
if os.path.exists(checkpoint_folder):
    os.remove(checkpoint_folder)
checkpointer = ModelCheckpoint(filepath=checkpoint_folder, monitor='val_loss',
                                   mode='min', save_best_only=True, verbose=1)
# Creating tensorboard to realtime evaluate the model
log_dir = "/home/lactec/dados/mestrado_gabriel/coloradar/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
# Create a learning rate schedule
lrate = LearningRateScheduler(step_decay)
loss_weights = {'fc_trans': 1,'fc_rot': 1}
input_shape = (32, 128, 128, 2)
input = Input(input_shape)
flownet3D_features = nets.build_3D_flownet(input)
fc_trans, fc_rot = nets.build_6D_pose_regressor(flownet3D_features)
model = Model(inputs=input, outputs=[fc_trans, fc_rot])
model.summary()
rms_prop = RMSprop(learning_rate=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rms_prop,
              loss={'fc_trans': 'mse', 'fc_rot': 'mse'},
              loss_weights=loss_weights)
history_model = model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=[lrate, checkpointer, tensorboard_callback], verbose=1)
history_file = open("/home/lactec/dados/mestrado_gabriel/coloradar/models/history.pkl", "wb")
pickle.dump(history_model, history_file)
