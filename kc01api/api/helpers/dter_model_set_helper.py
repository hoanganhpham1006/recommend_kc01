from django.conf import settings
import os
from api import dter_model, remap_info, map_url_trainid, tf, recall20
import pickle

def processing(dataset_name, model_name):
    if not os.path.isdir(os.path.join(settings.BASE_DIR, "api/models", dataset_name, model_name)):
        return False
    else:
        print("Restoring DTER Model .. " + dataset_name + "/" + model_name)
        model_path = settings.BASE_DIR + "/api/models/" + dataset_name + "/" + model_name
        data_path = settings.BASE_DIR + "/api/databases/" + dataset_name
        subfix = '_'.join(model_path.split('/')[-1].split('_')[1:])

        dter_model = tf.keras.models.load_model(model_path, custom_objects={"recall20": recall20}, compile=False)
        dter_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=[recall20])
        test_input = tf.keras.preprocessing.sequence.pad_sequences([[1, 1]], maxlen=20, dtype='int64', padding='pre', truncating='pre', value=0)
        dter_model.predict(test_input)
        print("Restored successfully DTER Model! " + dataset_name + "/" + model_name)
        map_url_trainid = pickle.load(open(data_path + '/map_url_trainid_' + subfix + '.pkl', 'rb'))
        remap_info = pickle.load(open(data_path + '/remap_info_' + subfix + '.pkl', 'rb'))
        with open(settings.BASE_DIR + "/api/logs/model_log.txt", "a") as f:
            f.write(dataset_name + "/" + model_name + "\n")
        return True