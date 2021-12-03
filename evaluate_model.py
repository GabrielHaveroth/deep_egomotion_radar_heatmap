import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from data_loader import *
import networks as nets
from helpers import *
results_path = "./results"
old_model = tf.keras.models.load_model('./models/weights-058-0.0320.hdf5')
df_data = pd.read_pickle('./metadata/test.pkl')
all_radar_params = get_cascade_params('/home/lactec/dados/mestrado_gabriel/calib/')
radar_heatmap_params = all_radar_params['heatmap']

input_type = '2d_cart_heatmap'

# To be faster
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
if input_type == '2d_cart_heatmap':
    input_shape = (128, 128, 2)
    input = Input(input_shape)
    teste_gen = RadarEgomotionDataGenerator(df_data,
                                            all_radar_params,
                                            batch_size=8,
                                            data_type=input_type, shuffle=False)

    flownet2D_features = nets.build_2D_flownet(input)
    fc_trans, fc_rot = nets.build_2D_pose_regressor(flownet2D_features)
    model = Model(inputs=input, outputs=[fc_trans, fc_rot])
    data = np.load('/home/lactec/dados/mestrado_gabriel/heatmap_test.npy')
    df_data = pd.read_pickle('./metadata/train.pkl')
    delta_poses = df_data['delta_poses_2D'].values.copy()
    delta_poses_arr = []
    heatmaps = np.load('/home/lactec/dados/mestrado_gabriel/test_heatmap.npy')

    for delta_p in delta_poses:
        delta_p = np.array(delta_p)
        delta_poses_arr.append(delta_p)
        
    delta_poses_arr = np.array(delta_poses_arr)
    delta_poses_arr = np.array(delta_poses_arr)
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    delta_poses_arr = np.asarray(delta_poses_arr)
    delta_poses_arr = scaler.transform(delta_poses_arr)
    y = delta_poses_arr
    X = heatmaps
    y_rot = delta_poses_arr[:, 2]
    y_trans = delta_poses_arr[:, 0:2]
    y = [y_trans, y_rot]
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss={'fc_trans': 'mse', 'fc_rot': 'mse'})
    model.set_weights(weights)
    y_pred = model.predict(X, y, batch_size=1, verbose=1)
    loss = model.evaluate(teste_gen, batch_size=1, verbose=1)

elif input_type == '3d_heatmap':
    input_shape = (32, 128, 128, 2)
    input = Input(input_shape)
    teste_gen = RadarEgomotionDataGenerator(df_data,
                                            all_radar_params,
                                            batch_size=8,
                                            data_type=input_type, shuffle=False)

    flownet3D_features = nets.build_3D_flownet(input)
    fc_trans, fc_rot = nets.build_3D_pose_regressor(flownet3D_features)
    model = Model(inputs=input, outputs=[fc_trans, fc_rot])
    rms_prop = Adam(learning_rate=0.001)
    model.compile(optimizer=rms_prop, loss={'fc_trans': 'mse', 'fc_rot': 'mse'})


print('-------Salving results--------')
with open('./results/poses.npy', 'wb') as f:
    np.save(f, y_pred[0])
with open('./results/angles.npy', 'wb') as f:
    np.save(f, y_pred[1])
with open('./results/loss.npy', 'wb') as f:
    np.save(f, np.array(loss))
