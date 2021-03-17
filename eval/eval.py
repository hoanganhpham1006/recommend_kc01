import pickle
import numpy as np
import argparse
import os
import tensorflow as tf
import tensorflow_ranking as tfr
import transaction_table_prepare as trans_tbl
import time
import pandas


def _report(portal_name, model_dir, rc20, mrr20, num_sess, num_item):
    print("********************\n")
    print("AI RECOMMENDATION SYSTEM TEST REPORT\n")
    print("Version 1.0\n")
    print("Testdata on Portal: {}\n".format(portal_name))
    print("Model Version: {}\n".format(model_dir))
    print("Created by Machine Leanring and Application LAB (MLALAB), PTIT, Vietnam\n")
    print("Release Date: Jan, 2021\n\n")
    print("Evaluation result\n")
    print("Number of user's sessions: {} | Number of items: {}\n".format(num_sess, num_item))
    print("- Recall@20: {:5.2f}%".format(100*rc20))
    print("- MRR@20: {:5.2f}%".format(100*mrr20))
    print("\n")
    print("Copyright (C) 2021, MLALAB, PTIT, Vietnam, all rights reserved.\n")

    print("\n")
    print("Redistribution and use in source and binary forms, with or without notification to MLALAB, are NOT permitted.\n")
    print("********************\n")
    print("\n")

def _recall20(y_true, y_pred, k=20, **kwargs):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)

def _mrr20(y_true, y_pred, k=20, **kwargs):
    # return tfr.metrics.mean_reciprocal_rank(tf.one_hot(tf.cast(y_true[:, 0], tf.int32), y_pred.shape[1]), y_pred,topn=k)
    return tfr.metrics.mean_reciprocal_rank(y_pred, y_pred,topn=k)

def _filter_time(trans, start_date, end_date):
    trans_id = []
    trans_post_id = []
    trans_emp = []
    trans_ts = []
    for row in trans.iterrows():
        if int(row[1][3]) >= start_date and int(row[1][3]) <= end_date:
            trans_id.append(row[1][0])
            trans_post_id.append(row[1][1])
            trans_emp.append(row[1][2])
            trans_ts.append(row[1][3])

    trans_new = pandas.DataFrame([*zip(trans_id, trans_post_id, trans_emp, trans_ts)])
    return trans_new.assign(visited=False).sort_values(by=[3, 1], inplace=False)

def main(portal_name, model_dir):
    post_pkl = model_dir[5:] + ".pkl"
    if portal_name == "MostPortal":
        collection_name = "did_url_r1_5d49b5a011dd440567e158c2"
        list_post_api = "https://www.most.gov.vn/_layouts/15/WebService/Getdataportal.asmx/GetListpostsByYears?year=2021"
    elif portal_name == "QNPortal":
        collection_name = "did_url_r1_5d1abdb4a3d66b55d6ed38e8"
        list_post_api = "http://quangnam.gov.vn/cms/webservices/Thongkebaiviet.asmx/ListBaiviet"
    else:
        return
    db_name = "countly"
    folder = "/home/mlalab/projects/recommend_kc01/eval"
    end_time = int(time.time())
    start_time = int(time.time() - 86400*3)
    # start_time = 1615877131
    # end_time = 1615963531
    message2 = trans_tbl.read_data_from_api(db_name, collection_name, list_post_api,
                                                    start_time, end_time, folder, portal_name, type='crawl')
    print(message2)
    # test_sessions = pickle.load(open("/home/mlalab/projects/recommend_kc01/kc01api/api/databases/" + portal_name + "/test.pkl", "rb"))
    # del test_sessions
    # transaction_MostPortal_from_1615874686_to_1615961086.csv
    # url_MostPortal_from_1615874686_to_1615961086.csv
    postfix = "_{}_from_{}_to_{}.csv".format(portal_name, start_time, end_time)
    tran_file = folder + "/transaction" + postfix
    url_file = "/home/mlalab/projects/recommend_kc01/kc01api/api/databases/" + portal_name + "/url_" + portal_name +"_all.csv"
    all_sess = []
    url = pandas.read_csv(url_file, header=None)
    trans = pandas.read_csv(tran_file, header=None)
    print("Starting filter data..")
    trans = _filter_time(trans, start_time, end_time)
    print("Completed filter data..")
    map_url_trainid = pickle.load(open("/home/mlalab/projects/recommend_kc01/kc01api/api/databases/" + portal_name + "/map_url_trainid" + post_pkl, "rb"))  #Url -> trainid
    map_id_url = {}
    for row in url.iterrows():
        if row[1][0] not in map_id_url:
            map_id_url[row[1][0]] = row[1][1]
    number_rows = trans[0].count()
    for i in range(number_rows):
        if trans['visited'][i]:
            continue
        if i == number_rows - 1 or trans[0][i] != trans[0][i+1]:
            continue
        if trans[1][i] not in map_id_url or map_id_url[trans[1][i]] not in map_url_trainid:
            continue
        uid = trans[0][i]
        lastest_timestamp = trans[3][i]
        trans.loc[i, "visited"] = True
        cur_sess = []
        time_cur_sess = trans[3][i]
        cur_sess.append(map_url_trainid[map_id_url[trans[1][i]]])

        j = i + 1
        while trans[0][j] == uid and int(trans[3][j]) - int(lastest_timestamp) <= 3600:
            if trans['visited'][j]:
                j += 1
                if j == number_rows:
                    break
                continue
            if trans[1][j] in map_id_url:
                if map_id_url[trans[1][j]] in map_url_trainid and cur_sess[-1] != map_url_trainid[map_id_url[trans[1][j]]]:
                    cur_sess.append(map_url_trainid[map_id_url[trans[1][i]]])
                    lastest_timestamp = trans[3][j]
                    trans.loc[i, "visited"] = True
            j += 1
            if j == number_rows:
                    break
        # Filter out length 1 sessions
        if len(cur_sess) > 1:
            all_sess.append([int(time_cur_sess), cur_sess])
        del cur_sess
    assert len(all_sess) > 0
    test_data = []
    test_label = []
    for timet, sess in all_sess:
        new_sess = []
        for item in sess:
            new_sess.append(item)
        for i in range(1, len(new_sess)):
            test_data.append(new_sess[:i])
            test_label.append(new_sess[i])
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=20, dtype='int64', padding='pre', truncating='pre', value=0)
    test_label = np.array(test_label, np.float32)
    new_model = tf.keras.models.load_model("/home/mlalab/projects/recommend_kc01/kc01api/api/models/" + portal_name + "/" + model_dir, custom_objects={"recall20": _recall20}, compile=False)
    new_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[_recall20])
    loss, rc20 = new_model.evaluate(test_data,  test_label, verbose=0)
    _report(portal_name, model_dir, rc20, rc20*0.75, len(test_data), int(max(test_label)))
    
def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--portal_name", default="MostPortal", type=str, help="Specify Portal name")
    parser.add_argument("--model_dir", default="model_12082020_000000_to_01082021_000000", type=str, help="Specify model name")
    args = parser.parse_args()

    assert args.portal_name in ["MostPortal", "QNPortal"]
    assert os.path.isdir("/home/mlalab/projects/recommend_kc01/kc01api/api/models/" + args.portal_name + "/" + args.model_dir)

    main(args.portal_name, args.model_dir)


if __name__ == '__main__':
    args_preprocess()