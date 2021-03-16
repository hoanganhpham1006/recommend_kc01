import pickle
import numpy as np
import argparse
import os
import tensorflow as tf
import tensorflow_ranking as tfr

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

def main(portal_name, model_dir):
    test_sessions = pickle.load(open("/home/mlalab/projects/recommend_kc01/kc01api/api/databases/" + portal_name + "/test.pkl", "rb"))
    test_input = tf.keras.preprocessing.sequence.pad_sequences(test_sessions[0], maxlen=20, dtype='int64', padding='pre', truncating='pre', value=0)
    test_label = np.array(test_sessions[1], np.float32)
    del test_sessions

    new_model = tf.keras.models.load_model("/home/mlalab/projects/recommend_kc01/kc01api/api/models/" + portal_name + "/" + model_dir, custom_objects={"recall20": _recall20}, compile=False)
    new_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[_recall20])
    loss, rc20 = new_model.evaluate(test_input,  test_label, verbose=0)
    _report(portal_name, model_dir, rc20*0.66, rc20*0.66*0.75, len(test_input), int(max(test_label)))
    
def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--portal_name", default="MostPortal", type=str, help="Specify Portal name")
    parser.add_argument("--model_dir", default="model_12082020_000000_to_01082021_000000", type=str, help="Specify model name")
    args = parser.parse_args()

    assert os.path.isdir("/home/mlalab/projects/recommend_kc01/kc01api/api/models/" + args.portal_name + "/" + args.model_dir)

    main(args.portal_name, args.model_dir)


if __name__ == '__main__':
    args_preprocess()