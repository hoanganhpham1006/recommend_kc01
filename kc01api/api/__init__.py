import tensorflow as tf
import pickle
from django.conf import settings
import os

status = {} #'MostPortal': 0, 'QNPortal': 0, 'PTIT': 0
lastest = {}
dter_model = {}
lastest_model_path = {}
lastest_data_path = {}
map_url_trainid = {}
remap_info = {}
supported_dataset = ['MostPortal', 'QNPortal']

def recall20(y_true, y_pred, k=20, **kwargs):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)

if os.path.isfile(settings.BASE_DIR + "/api/logs/model_log.txt"):
    with open(settings.BASE_DIR + "/api/logs/model_log.txt", "r") as f:
        lines = f.readlines()

    for l in lines:
        status[l.split('/')[0]] = 1
        lastest[l.split('/')[0]] = l[:-1]

    for dataset in lastest.keys():
        lastest_model_path[dataset] = settings.BASE_DIR + "/api/models/" + lastest[dataset]
        lastest_data_path[dataset] = '_'.join(lastest[dataset].split('/')[-1].split('_')[1:])

        print("Restoring DTER Model .. " + lastest[dataset])
        dter_model[dataset] = tf.keras.models.load_model(lastest_model_path[dataset], custom_objects={"recall20": recall20}, compile=False)

        dter_model[dataset].compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=[recall20])
        test_input = tf.keras.preprocessing.sequence.pad_sequences([[1, 1]], maxlen=20, dtype='int64', padding='pre', truncating='pre', value=0)
        dter_model[dataset].predict(test_input)
        print("Restored successfully DTER Model! " + lastest[dataset])

        map_url_trainid[dataset] = pickle.load(open(settings.BASE_DIR + "/api/databases/" + lastest[dataset].split('/')[0] + '/map_url_trainid_' + lastest_data_path[dataset] + '.pkl', 'rb'))
        remap_info[dataset] = pickle.load(open(settings.BASE_DIR + "/api/databases/" + lastest[dataset].split('/')[0] + '/remap_info_' + lastest_data_path[dataset] + '.pkl', 'rb'))
    
# #Statistic process
# most_popular_results = [l.split(' | ') for l in open(settings.BASE_DIR + '/mostpopular.txt', 'rt', encoding='utf-8').readlines()]
# hot_trend_results = [l.split(' | ') for l in open(settings.BASE_DIR + '/hottrend.txt', 'rt', encoding='utf-8').readlines()]
# stat_rec = []

# for i in range(10):
#     stat_rec.append([most_popular_results[i][0], most_popular_results[i][1][:-1], "Most-Popular"])
#     stat_rec.append([hot_trend_results[i][0], hot_trend_results[i][1][:-1], "Hot-Trend"])