from pathlib import Path
import os
import sys
import getopt
sys.path.append(str(Path(os.getcwd()).parent))
from api.helpers.common import *
from api.helpers.config import dataset_config
import api.helpers.category_table_prepare as cat_tbl
import api.helpers.transaction_table_prepare as trans_tbl
import api.helpers.post_table_prepare as post_tbl
from api.helpers.log_status_helper import logd, status_from_logd
from django.conf import settings
import os
import threading
import pandas
import pickle

import tensorflow as tf
import numpy as np
import unicodedata, re
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Layer, Embedding, LayerNormalization
import keras

from datetime import datetime, timedelta, timezone
import time

SPLIT_SESSION_SECOND = 3600
# Transformer parameters
d_model = 32 # 512 in the original paper
d_k = 16 # 64 in the original paper
d_v = 16 # 64 in the original paper

n_heads = 2 # 8 in the original paper
n_decoder_layers = 1 # 6 in the original paper

max_token_length = 20 # 512 in the original paper
# number_items = 37483 + 1 + 1
# number_items = 2668 + 1 #MOST

class SingleHeadAttention(Layer):
  def __init__(self, input_shape=(3, -1, d_model), dropout=.0, masked=None):
    super(SingleHeadAttention, self).__init__()
    self.q = Dense(d_k, input_shape=(-1, d_model), kernel_initializer='glorot_uniform', 
                   bias_initializer='glorot_uniform', name='q')
    self.normalize_q = Lambda(lambda x: x / np.sqrt(d_k))
    self.k = Dense(d_k, input_shape=(-1, d_model), kernel_initializer='glorot_uniform', 
                   bias_initializer='glorot_uniform', name='k')
    self.v = Dense(d_v, input_shape=(-1, d_model), kernel_initializer='glorot_uniform', 
                   bias_initializer='glorot_uniform', name='v')
    self.dropout = dropout
    self.masked = masked
  
  # Inputs: [query, key, value]
  def call(self, inputs, training=None):
    assert len(inputs) == 3
    # We use a lambda layer to divide vector q by sqrt(d_k) according to the equation
    q = self.normalize_q(self.q(inputs[0]))
    k = self.k(inputs[1])
    # The dimensionality of q is (batch_size, query_length, d_k) and that of k is (batch_size, key_length, d_k)
    # So we will do a matrix multication by batch after transposing last 2 dimensions of k
    # tf.shape(attn_weights) = (batch_size, query_length, key_length)
    attn_weights = tf.matmul(q, tf.transpose(k, perm=[0,2,1]))
    if self.masked: # Prevent future attentions in decoding self-attention
      # Create a matrix where the strict upper triangle (not including main diagonal) is filled with -inf and 0 elsewhere
      length = tf.shape(attn_weights)[-1]
      #attn_mask = np.triu(tf.fill((length, length), -np.inf), k=1) # We need to use tensorflow functions instead of numpy
      attn_mask = tf.fill((length, length), -np.inf)
      attn_mask = tf.linalg.band_part(attn_mask, 0, -1) # Get upper triangle
      attn_mask = tf.linalg.set_diag(attn_mask, tf.zeros((length))) # Set diagonal to zeros to avoid operations with infinity
      # This matrix is added to the attention weights so all future attention will have -inf logits (0 after softmax)
      attn_weights += attn_mask
    # Softmax along the last dimension
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)
    if training: # Attention dropout included in the original paper. This is possibly to encourage multihead diversity.
      attn_weights = tf.nn.dropout(attn_weights, rate=self.dropout)
    v = self.v(inputs[2])
    return tf.matmul(attn_weights, v)

class MultiHeadAttention(Layer):
  def __init__(self, dropout=.0, masked=None):
    super(MultiHeadAttention, self).__init__()
    self.attn_heads = list()
    self.masked = masked
    for i in range(n_heads): 
      self.attn_heads.append(SingleHeadAttention(dropout=dropout, masked=self.masked))
    self.linear = Dense(d_model, input_shape=(-1, n_heads * d_v), kernel_initializer='glorot_uniform', 
                   bias_initializer='glorot_uniform')
    
  def call(self, x, training=None):
    attentions = [self.attn_heads[i](x, training=training) for i in range(n_heads)]
    concatenated_attentions = tf.concat(attentions, axis=-1)
    return self.linear(concatenated_attentions)

# My decoder

class TransformerDecoder(Layer):
  def __init__(self, dropout=.0, attention_dropout=.0, masked=None, **kwargs):
    super(TransformerDecoder, self).__init__(**kwargs)
    self.dropout_rate = dropout
    self.attention_dropout_rate = attention_dropout
    self.masked = masked
  def build(self, input_shape):
    self.multihead_self_attention_1 = MultiHeadAttention(dropout=self.attention_dropout_rate, masked=self.masked)
    self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
    self.layer_normalization1 = LayerNormalization(input_shape=input_shape, epsilon=1e-6)
    
    self.multihead_self_attention_2 = MultiHeadAttention(dropout=self.attention_dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
    self.layer_normalization2 = LayerNormalization(input_shape=input_shape, epsilon=1e-6)
    
    self.linear1 = Dense(input_shape[-1] * 4, input_shape=input_shape, activation='relu',
                        kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
    self.linear2 = Dense(input_shape[-1], input_shape=self.linear1.compute_output_shape(input_shape),
                        kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
    self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)
    self.layer_normalization3 = LayerNormalization(input_shape=input_shape, epsilon=1e-6)
    super(TransformerDecoder, self).build(input_shape)
  def call(self, x, training=None):
    sublayer1 = self.multihead_self_attention_1((x, x, x))
    sublayer1 = self.dropout1(sublayer1, training=training)
    layernorm1 = self.layer_normalization1(x + sublayer1)
    
    # sublayer2 = self.multihead_self_attention_2((layernorm1, layernorm1, layernorm1))
    # sublayer2 = self.dropout2(sublayer2, training=training)
    # layernorm2 = self.layer_normalization2(layernorm1 + sublayer2)
    
    # sublayer3 = self.linear2(self.linear1(layernorm2))
    # sublayer3 = self.dropout3(sublayer3, training=training)
    # layernorm3 = self.layer_normalization2(layernorm2 + sublayer3)

    sublayer3 = self.linear2(self.linear1(layernorm1))
    sublayer3 = self.dropout3(sublayer3, training=training)
    layernorm3 = self.layer_normalization2(layernorm1 + sublayer3)
    return layernorm3
  def compute_output_shape(self, input_shape):
    return input_shape

class SinusoidalPositionalEncoding(Layer): #Inspired from https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer_Torch.ipynb
  def __init__(self):
    super(SinusoidalPositionalEncoding, self).__init__()
    self.sinusoidal_encoding = np.array([self.get_positional_angle(pos) for pos in range(max_token_length)], dtype=np.float32)
    self.sinusoidal_encoding[:, 0::2] = np.sin(self.sinusoidal_encoding[:, 0::2])
    self.sinusoidal_encoding[:, 1::2] = np.cos(self.sinusoidal_encoding[:, 1::2])
    self.sinusoidal_encoding = tf.cast(self.sinusoidal_encoding, dtype=tf.float32) # Casting the array to Tensor for slicing
  def call(self, x):
    return x + self.sinusoidal_encoding[:tf.shape(x)[1]]
  def compute_output_shape(self, input_shape):
    return input_shape
  def get_angle(self, pos, dim):
    return pos / np.power(10000, 2 * (dim // 2) / d_model)
  def get_positional_angle(self, pos):
    return [self.get_angle(pos, dim) for dim in range(d_model)]

class DTER(Model):
  def __init__(self, dropout=.1, attention_dropout=.0, masked=None, number_items=0, **kwargs):
    super(DTER, self).__init__(**kwargs)
    self.embedding_mat = tf.Variable(tf.random.uniform([number_items, d_model], -1.0, 1.0), trainable=True)
    self.pos_encoding = SinusoidalPositionalEncoding()
    self.global_decoder = [TransformerDecoder(dropout=dropout, attention_dropout=attention_dropout, masked=masked) for i in range(n_decoder_layers)]
    self.local_decoder = [TransformerDecoder(dropout=dropout, attention_dropout=attention_dropout, masked=masked) for i in range(n_decoder_layers)]
    self.local_decoder_special = [TransformerDecoder(dropout=dropout, attention_dropout=attention_dropout, masked=masked) for i in range(n_decoder_layers)]
    # self.ht_s = tf.Variable(tf.random.uniform([1, d_model], -1.0, 1.0), trainable=True)
    self.A1 =  tf.Variable(tf.random.uniform([d_model, d_model], -1.0, 1.0), trainable=True)
    self.A2 =  tf.Variable(tf.random.uniform([d_model, d_model], -1.0, 1.0), trainable=True)
    self.vT = tf.Variable(tf.random.uniform([1, d_model], -1.0, 1.0), trainable=True)
    # self.A1_s =  tf.Variable(tf.random.uniform([d_model, d_model], -1.0, 1.0), trainable=True)
    # self.A2_s =  tf.Variable(tf.random.uniform([d_model, d_model], -1.0, 1.0), trainable=True)
    # self.vT_s = tf.Variable(tf.random.uniform([1, d_model], -1.0, 1.0), trainable=True)
    self.B = tf.Variable(tf.random.uniform((2*d_model, d_model), -1.0, 1.0), trainable=True)

  def call(self, inputs, training=None): # Source_sentence and decoder_input
    source_session = inputs
    embedded_source = tf.nn.embedding_lookup(self.embedding_mat, source_session)
    encoder_output = self.pos_encoding(embedded_source)

    global_embedding = encoder_output
    local_embedding = encoder_output
    local_embedding_s = encoder_output

    for global_decoder_unit in self.global_decoder:
      global_embedding = global_decoder_unit(global_embedding, training=training)
    global_embedding = global_embedding[:, -1, :]

    for local_decoder_unit in self.local_decoder:
      local_embedding = local_decoder_unit(local_embedding, training=training)
    ht = tf.reduce_mean(local_embedding, axis=1)
    # ht = local_embedding[:, -1, :]
    ht = tf.tile(tf.expand_dims(ht, -1), (1, 1, max_token_length))

    hj = tf.transpose(local_embedding, [0, 2, 1])

    attention_weights = tf.keras.activations.hard_sigmoid(tf.matmul(self.A1, hj) + tf.matmul(self.A2, ht))
    attention_weights = tf.matmul(self.vT, attention_weights)

    local_embedding = tf.math.divide(tf.matmul(attention_weights, local_embedding)[:, 0, :], tf.keras.backend.sum(attention_weights, axis=2))

    # for local_decoder_s_unit in self.local_decoder_special:
    #   local_embedding_s = local_decoder_s_unit(local_embedding_s, training=training)
    # ones_like = tf.ones_like(tf.reduce_mean(local_embedding_s, axis=1))
    # # ht = local_embedding[:, -1, :]
    # ht_s = self.ht_s*ones_like
    # ht_s = tf.tile(tf.expand_dims(ht_s, -1), (1, 1, max_token_length))

    # hj_s = tf.transpose(local_embedding_s, [0, 2, 1])
    # attention_weights_s = tf.keras.activations.hard_sigmoid(tf.matmul(self.A1_s, hj_s) + tf.matmul(self.A2_s, ht_s))
    # attention_weights_s = tf.matmul(self.vT_s, attention_weights_s)

    # local_embedding_s = tf.math.divide(tf.matmul(attention_weights_s, local_embedding_s)[:, 0, :], tf.keras.backend.sum(attention_weights_s, axis=2))

    c_T = tf.concat([global_embedding, local_embedding], -1)
    emb_T = tf.transpose(self.embedding_mat) # (d_model, vocab_size)
    weights = tf.matmul(self.B, emb_T)
    logits = tf.matmul(c_T, weights)

    decoder_output = tf.nn.softmax(logits, axis=-1)
    return decoder_output

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 2 == 0:
            if epoch == 79:
                logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 60 + (epoch + 1)//2 - 1, "Training End")
            else:
                logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 60 + (epoch + 1)//2, "Training..")



#Training
def recall20(y_true, y_pred, k=20, **kwargs):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)

def thread_function(dataset_name, start_date, end_date):
    crawl_success = crawl(dataset_name, start_date, end_date)
    if not crawl_success:
        logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", -1, "Crawl Error")
        return False
    preprocess_sucess, number_items = preprocess(dataset_name, start_date, end_date)
    if not preprocess_sucess:
        logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", -1, "Preprocess Error")
        return False
    train_success = train_dter(dataset_name, start_date, end_date, number_items)
    if not train_success:
        logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", -1, "Training Error")
        return False
    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 100, "Training end")
    return True

def train_dter(dataset_name, start_date, end_date, number_items):
    if dataset_name == "MostPortal":
        dataset_name = "Most_Portal"
    elif dataset_name == "QNPortal":
        dataset_name = "QN_Portal"

    start_str = datetime.fromtimestamp(start_date).strftime("%m%d%Y_%H%M%S")
    end_str = datetime.fromtimestamp(end_date).strftime("%m%d%Y_%H%M%S")

    with open(settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/train.pkl", 'rb') as f:
        train_sessions = pickle.load(f)
    
    with open(settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/test.pkl", 'rb') as f:
        test_sessions = pickle.load(f)

    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 55, "Done trainer load packed data")
    
    train_input = tf.keras.preprocessing.sequence.pad_sequences(train_sessions[0], maxlen=max_token_length, dtype='int64', padding='pre', truncating='pre', value=0)
    train_label = np.array(train_sessions[1], np.float32)
    test_input = tf.keras.preprocessing.sequence.pad_sequences(test_sessions[0], maxlen=max_token_length, dtype='int64', padding='pre', truncating='pre', value=0)
    test_label = np.array(test_sessions[1], np.float32)

    del train_sessions, test_sessions

    dter_keras = DTER(dropout=0.1, attention_dropout=0.1, masked=True, number_items=number_items)
    opt = keras.optimizers.Adam(learning_rate=0.007)

    dter_keras.compile(optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=[recall20])

    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 60, "Done trainer prepare")   
    
    dter_keras.fit(train_input, train_label, \
               verbose=1, batch_size=64, epochs=80, \
               validation_data=(test_input, test_label),
               callbacks=[CustomCallback()])
    
    dter_keras.save(settings.BASE_DIR + "/api/models/" + dataset_config[dataset_name]["dataset"] + "/model_" + start_str + "_to_" + end_str)
    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 100, "Done saving model!")
    return True
    

def preprocess(dataset_name, start_date, end_date):
    if dataset_name == "MostPortal":
        dataset_name = "Most_Portal"
    elif dataset_name == "QNPortal":
        dataset_name = "QN_Portal"

    if not dataset_config[dataset_name]["folder"]:
        os.mkdir(dataset_config[dataset_name]["folder"]) 

    start_str = datetime.fromtimestamp(start_date).strftime("%m%d%Y_%H%M%S")
    end_str = datetime.fromtimestamp(end_date).strftime("%m%d%Y_%H%M%S")
    ext_trans = "_from_" + str(start_date) + "_to_" + str(end_date) + ".csv"
    cat_file = settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/cat_" + dataset_config[dataset_name]["dataset"] +  "_all.csv"
    post_file = settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/post_" + dataset_config[dataset_name]["dataset"] + "_all.csv"
    tran_file = settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/transaction_" + dataset_config[dataset_name]["dataset"]  + ext_trans
    url_file = settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/url_" + dataset_config[dataset_name]["dataset"] + ext_trans
    all_sess = []

    url = pandas.read_csv(url_file, header=None)
    trans = pandas.read_csv(tran_file, header=None).assign(visited=False).sort_values(by=[3, 1], inplace=False)
    posts = pandas.read_csv(post_file, header=None)

    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 20, "Done Read CSV")

    map_id_title = {}
    for row in posts.iterrows():
        if row[1][0] not in map_id_title:
            map_id_title[row[1][0]] = row[1][1]
    map_id_nid = {}
    map_url_nid = {}
    map_nid_info = {}

    row_i = 1
    for row in url.iterrows():
        if row[1][0] not in map_id_nid:
            map_id_nid[row[1][0]] = row_i
            map_url_nid[row[1][1]] = row_i
            
            title = ''
            if row[1][0] in map_id_title:
                title = map_id_title[row[1][0]]
            
            map_nid_info[row_i] = [row[1][1], title]
            row_i += 1

    number_rows = trans[0].count()
    for i in range(number_rows):
        if trans['visited'][i]:
            continue
        if i == number_rows - 1 or trans[0][i] != trans[0][i+1]:
            continue
        if trans[1][i] not in map_id_nid:
            continue
        uid = trans[0][i]
        lastest_timestamp = trans[3][i]
        trans.loc[i, "visited"] = True
        cur_sess = []
        time_cur_sess = trans[3][i]
        
    #     if result[i]['itemid'] not in re_id_item:
    #         re_id_item[result[i]['itemid']] = item_no
    #         item_no += 1
        cur_sess.append(map_id_nid[trans[1][i]])
        j = i + 1
        while trans[0][j] == uid and int(trans[3][j]) - int(lastest_timestamp) <= SPLIT_SESSION_SECOND:
            if trans['visited'][j]:
                j += 1
                if j == number_rows:
                    break
                continue
            if trans[1][j] in map_id_nid and cur_sess[-1] != map_id_nid[trans[1][j]]:
                cur_sess.append(map_id_nid[trans[1][j]])
                lastest_timestamp = trans[3][j]
                trans.loc[i, "visited"] = True
            j += 1
            if j == number_rows:
                    break
        # Filter out length 1 sessions
        if len(cur_sess) > 1:
            all_sess.append([int(time_cur_sess), cur_sess])
        del cur_sess

    maxtime = trans.iloc[-1, 3]

    SPLIT_TEST_TIMESTAMP = maxtime - 86400 * 1
    
    tra_sess = list(filter(lambda x: x[0] < SPLIT_TEST_TIMESTAMP, all_sess))
    tes_sess = list(filter(lambda x: x[0] >= SPLIT_TEST_TIMESTAMP, all_sess))

    train_data = []
    train_label = []
    remap_item = {} #nid -> train_id

    remap_info = {}
    map_url_trainid = {}

    item_no = 1
    for time, sess in tra_sess:
        new_sess = []
        for item in sess:
            if item not in remap_item:
                remap_item[item] = item_no
                remap_info[item_no] = map_nid_info[item]
                map_url_trainid[map_nid_info[item][0]] = item_no
                item_no += 1
            new_sess.append(remap_item[item])
        for i in range(1, len(new_sess)):
            train_data.append(new_sess[:i])
            train_label.append(new_sess[i])

    test_data = []
    test_label = []
    for time, sess in tes_sess:
        new_sess = []
        for item in sess:
            if item in remap_item:
                new_sess.append(remap_item[item])
        for i in range(1, len(new_sess)):
            test_data.append(new_sess[:i])
            test_label.append(new_sess[i])

    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 30, "Done process from CSV")
    print("Data processed: Train: " + str(len(train_data)) + ", Test: " + str(len(test_data)))
    
    with open(settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/train.pkl", "wb") as f:
        pickle.dump([train_data, train_label], f)
    del train_data, train_label
    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 35, "")
    with open(settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/test.pkl", 'wb') as f:
        pickle.dump([test_data, test_label], f)
    del test_data, test_label
    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 40, "")
    with open(settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/remap_info_" + start_str + "_to_" + end_str  + ".pkl", 'wb') as f:
        pickle.dump(remap_info, f)
    del remap_info
    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 45, "")
    with open(settings.BASE_DIR + "/api/databases/" + dataset_config[dataset_name]["dataset"] + "/map_url_trainid_" + start_str + "_to_" + end_str + ".pkl", 'wb') as f:
        pickle.dump(map_url_trainid, f)
    del map_url_trainid
    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 50, "Done Preprocess")
    
    number_items = len(map_url_nid)
    print("NUMBER ITEMS: " + str(number_items))
    return True, number_items

def crawl(dataset_name, start_date=None, end_date=None):
    if dataset_name == "MostPortal":
        dataset_name = "Most_Portal"
    elif dataset_name == "QNPortal":
        dataset_name = "QN_Portal"

    cfg = dataset_config[dataset_name]
    message1 = cat_tbl.read_data_from_api(cfg["list_cat_api"], cfg["folder"], cfg["dataset"])
    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 5, message1)
    message2 = trans_tbl.read_data_from_api(cfg["db_name"], cfg["collection_name"], cfg["list_post_api"],
                                                start_date, end_date, cfg["folder"], cfg["dataset"], type='crawl')
    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 10, message2)
    message3 = post_tbl.read_data_from_api(cfg["list_post_api"], None, cfg["folder"], cfg["dataset"],
                                               s_time=None, e_time=None, type='crawl')
    logd(settings.BASE_DIR + "/api/logs/train_log.txt", "a", 15, message3)
    return True

def processing(dataset_name, start_date, end_date, force_train):
    if end_date is None:
        end_date = int(time.time())
    if start_date is None:
        start_date = end_date - 86400*30
    if start_date > end_date:
        t = start_date
        start_date = end_date
        end_date = t
    model_existed = False

    start_str = datetime.fromtimestamp(start_date).strftime("%m%d%Y_%H%M%S")
    end_str = datetime.fromtimestamp(end_date).strftime("%m%d%Y_%H%M%S")
    if os.path.isdir(settings.BASE_DIR + "/api/models/" + dataset_name +"/model_" + str(start_date) + "_to_" + str(end_date)):
        model_existed = True

    if not model_existed or (model_existed and force_train):
        log_file = settings.BASE_DIR + "/api/logs/train_log.txt"
        if os.path.isfile(log_file):
            status, progress = status_from_logd(log_file)
            if status == '0':
                return "Training process has not finished! Please check status by another api"
            elif status == '-1':
                return "Training server are not working right, please check with admin for more information"
        logd(settings.BASE_DIR + "/api/logs/train_log.txt", "w", 0, "")
        x = threading.Thread(target=thread_function, args=(dataset_name, start_date, end_date))
        x.start()
        return "Training process began!"
    else:
        return "Model " + str(start_date) + "_to_" + str(end_date) + " is existed! Use force_train = True if you want to retrain"