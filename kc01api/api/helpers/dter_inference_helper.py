from api import dter_model, remap_info, map_url_trainid, tf
import numpy as np

def processing(list_url, dataset_name):
    list_url = [map_url_trainid[dataset_name][url] for url in list_url.split() if url in map_url_trainid[dataset_name]]
    print(list_url)
    list_url = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(list_url, 0), maxlen=20, dtype='int64', padding='pre', truncating='pre', value=0)
    softmax = dter_model[dataset_name].predict(list_url)
    results = softmax[0].argsort()[-50:][::-1]
    results = [r for r in list(results) if r in remap_info[dataset_name]]
    print(results)
    list_rec = [remap_info[dataset_name][nid] for nid in results if (nid != 0 and nid != list_url[0][-1])]
    return list_rec[:20]