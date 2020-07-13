import tensorflow as tf
import pickle
from django.conf import settings

def recall20(y_true, y_pred, k=20, **kwargs):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)

with open(settings.BASE_DIR + "/api/logs/model_log.txt", "r") as f:
    lastest = f.readlines()[-1][:-1]
lastest_model_path = settings.BASE_DIR + "/api/models/" + lastest
lastest_data_path = '_'.join(lastest.split('/')[-1].split('_')[1:])

print("Restoring DTER Model .. " + lastest)
dter_model = tf.keras.models.load_model(lastest_model_path, custom_objects={"recall20": recall20}, compile=False)

dter_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[recall20])
test_input = tf.keras.preprocessing.sequence.pad_sequences([[1, 1]], maxlen=20, dtype='int64', padding='pre', truncating='pre', value=0)
dter_model.predict(test_input)
print("Restored successfully DTER Model! " + lastest)

map_url_trainid = pickle.load(open(settings.BASE_DIR + "/api/databases/" + lastest.split('/')[0] + '/map_url_trainid_' + lastest_data_path + '.pkl', 'rb'))
remap_info = pickle.load(open(settings.BASE_DIR + "/api/databases/" + lastest.split('/')[0] + '/remap_info_' + lastest_data_path + '.pkl', 'rb'))
    
# #Statistic process
# most_popular_results = [l.split(' | ') for l in open(settings.BASE_DIR + '/mostpopular.txt', 'rt', encoding='utf-8').readlines()]
# hot_trend_results = [l.split(' | ') for l in open(settings.BASE_DIR + '/hottrend.txt', 'rt', encoding='utf-8').readlines()]
# stat_rec = []

# for i in range(10):
#     stat_rec.append([most_popular_results[i][0], most_popular_results[i][1][:-1], "Most-Popular"])
#     stat_rec.append([hot_trend_results[i][0], hot_trend_results[i][1][:-1], "Hot-Trend"])