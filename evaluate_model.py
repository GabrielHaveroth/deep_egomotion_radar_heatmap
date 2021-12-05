import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from data_loader import *
import networks as nets
from helpers import *
from tf_record_utils import get_dataset
results_path = "./results"
old_model = tf.keras.models.load_model('/home/lactec/dados/mestrado_gabriel/coloradar/models/weights-085-0.0056.hdf5')
df_data = pd.read_pickle('./metadata/test.pkl')
all_radar_params = get_cascade_params('/home/lactec/dados/mestrado_gabriel/calib/')
radar_heatmap_params = all_radar_params['heatmap']
input_type = '3d_heatmap'


weights = old_model.get_weights()
if input_type == '2d_cart_heatmap':
    input_shape = (128, 128, 2)
    input = Input(input_shape)
    # teste_gen = RadarEgomotionDataGenerator(df_data,
    #                                         all_radar_params,
    #                                         batch_size=8,
    #                                         data_type=input_type, shuffle=False)


    df_data = pd.read_pickle('./metadata/test.pkl')
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
    # delta_poses_arr = scaler.transform(delta_poses_arr)
    X = heatmaps
    y_rot = delta_poses_arr[:, 2]
    y_trans = delta_poses_arr[:, 0:2]
    y = [y_trans, y_rot]
    
    print(y_trans[0:10])

    # Bulding model
    flownet2D_features = nets.build_2D_flownet(input)
    fc_trans, fc_rot = nets.build_2D_pose_regressor(flownet2D_features)
    model = Model(inputs=input, outputs=[fc_trans, fc_rot])
    model.set_weights(weights)
    model.compile(loss={'fc_trans': 'mse', 'fc_rot': 'mse'})
    y_pred = model.predict(X, batch_size=1)
    loss = model.evaluate(X, y, batch_size=1, verbose=1)
    print(y_pred[0][0:10])

elif input_type == '3d_heatmap':
    test_dataset = get_dataset('test.tfrecords')
    input_shape = (32, 128, 128, 2)
    input = Input(input_shape)
    flownet3D_features = nets.build_3D_flownet(input)
    fc_trans, fc_rot = nets.build_6D_pose_regressor(flownet3D_features)
    model = Model(inputs=input, outputs=[fc_trans, fc_rot])
    model.compile(loss={'fc_trans': 'mse', 'fc_rot': 'mse'})
    model.set_weights(weights)
    y_pred = model.predict(test_dataset, verbose=1)
    loss = model.evaluate(test_dataset, verbose=1)


print('-------Salving results--------')
with open('./results/poses.npy', 'wb') as f:
    np.save(f, y_pred[0])
with open('./results/angles.npy', 'wb') as f:
    np.save(f, y_pred[1])
# with open('./results/loss.npy', 'wb') as f:
    # np.save(f, np.array(loss))
