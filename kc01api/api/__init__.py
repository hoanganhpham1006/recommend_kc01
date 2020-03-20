import tensorflow as tf
import pickle

def recall20(y_true, y_pred, k=20, **kwargs):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)

print("Restoring DTER Model ..")
dter_model = tf.keras.models.load_model('most_19Feb', custom_objects={"recall20": recall20}, compile=False)

dter_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[recall20])
test_input = tf.keras.preprocessing.sequence.pad_sequences([[1, 1]], maxlen=20, dtype='int64', padding='pre', truncating='pre', value=0)
dter_model.predict(test_input)
print("Restored successfully DTER Model!")

map_url_trainid = pickle.load(open('map_url_trainid.pkl', 'rb'))
remap_info = pickle.load(open('remap_info_most.pkl', 'rb'))