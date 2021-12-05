from tf_record_utils import save_heatmap_tfrecord
import pandas as pd

df_data = pd.read_pickle('./metadata/test.pkl')
items = df_data.values.tolist()
test_record_file = 'test.tfrecords'

save_heatmap_tfrecord(test_record_file, items)