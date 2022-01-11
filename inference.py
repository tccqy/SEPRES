"""
An example of inference
"""
import os
import lightgbm as lgb
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle


data_file_path = 'data'
model_file_path = 'models'
special_list = ['urineoutput_24h', 'net_balance', 'antibiotic_12h', 'antibiotic_24h', 'antibiotic_48h',
                'vent', 'sofa_24hours', 'age', 'weight', 'sus_infection', 'sn', 'label_sample', 'label_t',
                'label_sofa', 'positive_sample', 'icustay_id', 'data_len', 'hr', 'starttime', 'endtime',
                'valid_data']
more_list = ['max_map', 'min_map', 'max_hr', 'min_hr', 'max_sbp', 'min_sbp', 'max_dbp', 'min_dbp',
             'max_resp', 'min_resp', 'max_temp', 'min_temp', 'max_pao2', 'min_pao2', 'max_fio2',
             'min_fio2', 'max_spo2', 'min_spo2', 'max_tidal_volume', 'min_tidal_volume',
             'max_peak_insp_pressure', 'min_peak_insp_pressure', 'max_fibrinogen', 'min_fibrinogen']


def build_mlp_model(input_x, path):
    reader = tf.train.NewCheckpointReader(path)
    w1 = reader.get_tensor('w1')
    b1 = reader.get_tensor('b1')
    w2 = reader.get_tensor('w2')
    b2 = reader.get_tensor('b2')
    w3 = reader.get_tensor('w3')
    b3 = reader.get_tensor('b3')
    w4 = reader.get_tensor('w4')
    b4 = reader.get_tensor('b4')
    w5 = reader.get_tensor('w5')
    b5 = reader.get_tensor('b5')
    w6 = reader.get_tensor('w6')
    b6 = reader.get_tensor('b6')
    layer1 = tf.nn.relu(tf.matmul(input_x, w1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
    layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
    layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)
    layer5 = tf.nn.relu(tf.matmul(layer4, w5) + b5)
    output = tf.matmul(layer5, w6) + b6
    label = tf.nn.softmax(output)
    label1, label2 = tf.split(label, 2, 1)

    return label1


def read_data(file):
    df = pd.read_csv(os.path.join('.', data_file_path, file))
    cls_drop = ['sn', 'positive_sample', 'icustay_id', 'hr', 'starttime', 'endtime', 'sus_infection', 'valid_data']
    col = df.columns.values.tolist()
    remove = ['gcs', 'calcium', 'bnp', 'crp']
    for feature in col:
        for item in remove:
            if item in feature:
                cls_drop.append(feature)
        if 'avg' not in feature and feature not in special_list and feature not in more_list:
            cls_drop.append(feature)
    inputs = df.drop(columns=cls_drop)
    inputs = np.reshape(inputs.to_numpy(), (1, -1))

    return inputs


def pred_from_file(file, t):
    cls_data = read_data(file)
    scaler = pickle.load(open(os.path.join('.', model_file_path, 'lgb_%d.scalar.pkl' % t), 'rb'))
    cls_data = scaler.transform(cls_data)

    bst = lgb.Booster(params={'objective': 'binary'},
                      model_file=os.path.join('.', model_file_path, 'lgb_%dh_v4_rj.model' % t))
    preds_lgb = bst.predict(cls_data)[0]

    input_x = tf.placeholder(dtype=tf.float32, shape=[None, cls_data.shape[1]])
    label = build_mlp_model(input_x, os.path.join('.', model_file_path, 'mlp_tr_%dh_models' % t, 'mlp_tr_v4_%dh' % t))

    with tf.Session() as sess:
        preds_mlp = sess.run(label, feed_dict={input_x: cls_data})[0][0]

    return (preds_lgb + preds_mlp) / 2


if __name__ == '__main__':
    for t in range(1, 6):
        print('%dh Confidence Index:\t' % t, pred_from_file('data_example.csv', t))
