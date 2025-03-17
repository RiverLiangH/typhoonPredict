import tensorflow as tf
import pickle
import numpy as np

def decode_frame_ID(frame_ID_ascii):
    first_row = frame_ID_ascii[0]
    underscore_index = np.where(first_row == 95)[0][0]
    ID_bytes = first_row[:underscore_index]
    ID_bytes = bytes(ID_bytes)
    ID_str = ID_bytes.decode('utf-8')
    # scaler = scaler_dict.get(ID_bytes, None)
    return ID_str

def inverse_normalize(prediction, frame_ID):
    with open('vmax_scaler_dict.pkl', 'rb') as f:
        vmax_scaler_dict = pickle.load(f)
    vmax_scaler = vmax_scaler_dict[frame_ID]

    intensity_pred = vmax_scaler.inverse_transform(tf.reshape(prediction, (-1, 1))).flatten()
    return intensity_pred

def calculate_metric_dict(model, dataset):
    mae = tf.constant([0.])
    mse = tf.constant([0.])
    num = 0.

    for image_sequences, labels, feature, frame_ID_ascii, dV in dataset:
        raw_pred = model(image_sequences, feature, training=False)
        pred = inverse_normalize(raw_pred, decode_frame_ID(frame_ID_ascii))
        labels = inverse_normalize(labels, decode_frame_ID(frame_ID_ascii))
        labels = tf.cast(labels, tf.float32)
        pred = tf.cast(pred, tf.float32)

        sample_weight = tf.math.tanh((dV - 20) / 10) * 1000 + 1000.1
        sample_weight = tf.cast(sample_weight, tf.float32)

        mae_each = tf.math.reduce_mean((tf.abs(labels - pred)) * sample_weight)
        mse_each = tf.math.reduce_mean((tf.abs(labels - pred) ** 2) * sample_weight)

        num += 1
        mae = tf.add(mae, mae_each)
        mse = tf.add(mse, mse_each)

    MAE = tf.reduce_mean(tf.math.divide(mae, num))
    MSE = tf.reduce_mean(tf.math.divide(mse, num))

    return dict(
        MAE=MAE,
        MSE=MSE
    )