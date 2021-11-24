import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import RMSprop
from data_loader import *
import networks as nets
from helpers import *
model_path = ""
results_path = "./results"
old_model = tf.keras.models.load_model('/data/weights-046-0.0050.hdf5')
df_data = pd.read_pickle('./models/test.pkl')
all_radar_params = get_cascade_params('/data/Conjuntos_Dados_Mestrado/calib')
radar_heatmap_params = all_radar_params['heatmap']
teste_gen = RadarEgomotionDataGenerator(df_data,
                                        all_radar_params,
                                        batch_size=1,
                                        data_type='3d_heatmap', shuffle=False)


# poses_seq = dict()
# names = list(df_data['file'].unique())
# with mp.Pool(mp.cpu_count()) as pool:
#     results = [pool.apply_async(get_heatmap_poses, args=(name, radar_heatmap_params)) for name in names]
#     for r, name in zip(results, names):
#         poses_seq[name] = r.get()

# delta_poses, hm_powers_t12 = get_data_3D_batch_gt(df_data, radar_heatmap_params, poses_seq)
# delta_poses = np.asarray(delta_poses)
# y_batch_trans = delta_poses[:, 3:].copy()
# y_batch_rot = delta_poses[:, 0:3].copy()
# X_batch_power_heatmap = np.array(hm_powers_t12)
# X = X_batch_power_heatmap[0:80]
# y = [y_batch_trans[0:80], y_batch_rot[0:80]]
weights = old_model.get_weights()
input_shape = (32, 128, 128, 2)
input = Input(input_shape)
flownet3D_features = nets.build_3D_flownet(input)
fc_trans, fc_rot = nets.build_6D_pose_regressor(flownet3D_features)
model = Model(inputs=input, outputs=[fc_trans, fc_rot])
rms_prop = RMSprop(learning_rate=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rms_prop, loss={'fc_trans': 'mse', 'fc_rot': 'mse'})
model.set_weights(weights)
y_pred = model.predict(teste_gen, batch_size=1, verbose=1)
# loss = model.evaluate(teste_gen, batch_size=1, verbose=1)
print('-------Salving results--------')
with open('poses.npy', 'wb') as f:
    np.save(f, y_pred[0])
with open('angles.npy', 'wb') as f:
    np.save(f, y_pred[1])
with open('loss.npy', 'wb') as f:
    np.save(f, np.array(loss))
