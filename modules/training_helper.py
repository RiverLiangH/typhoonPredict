import tensorflow as tf
import folium
import time
import os
from modules.tfrecord_generator import coastline
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def decode_frame_ID(frame_ID_ascii):
    first_row = frame_ID_ascii[0]
    underscore_index = np.where(first_row == 95)[0][0]
    ID_bytes = first_row[:underscore_index]
    ID_bytes = bytes(ID_bytes)
    ID_str = ID_bytes.decode('utf-8')
    # scaler = scaler_dict.get(ID_bytes, None)
    return ID_str

def inverse_normalize(prediction, frame_ID):
    with open('TCSA_data/lon_scaler_dict.pkl', 'rb') as lon_f:
        lon_scaler_dict = pickle.load(lon_f)

    with open('TCSA_data/lat_scaler_dict.pkl', 'rb') as lat_f:
        lat_scaler_dict = pickle.load(lat_f)
    lon_scaler = lon_scaler_dict[frame_ID]
    lat_scaler = lat_scaler_dict[frame_ID]

    intensity_pred = prediction[:, 0]
    lon_pred_normalized = prediction[:, 1]
    lat_pred_normalized = prediction[:, 2]

    lon_pred = lon_scaler.inverse_transform(tf.reshape(lon_pred_normalized, (-1, 1))).flatten()
    lat_pred = lat_scaler.inverse_transform(tf.reshape(lat_pred_normalized, (-1, 1))).flatten()
    # intensity_pred = scaler.inverse_transform(tf.reshape(intensity_pred_normalized, (-1, 1))).flatten()

    prediction_inverse_normalized = np.column_stack((intensity_pred, lon_pred, lat_pred))
    return prediction_inverse_normalized

def calculate_metric_dict(model, dataset, draw_path = "None"):
    int_mae = tf.constant([0.])
    int_mse = tf.constant([0.])
    tck_mae = tf.constant([0.])
    tck_mse = tf.constant([0.])
    glo_mae = tf.constant([0.])
    glo_mse = tf.constant([0.])
    num = 0.

    def weighted_mse(p1, p2):
        squared_diff = tf.square(p1 - p2)
        mse = tf.reduce_sum(squared_diff)
        return mse

    def weighted_mae(p1, p2):
        abs_diff = tf.abs(p1 - p2)
        abs_diff_sum = tf.reduce_sum(abs_diff)
        return abs_diff_sum

    def draw_path_map(pred, true, save_path=None):
        print("PAINTING PATH MAP")
        m = folium.Map(location=[(max(pred[:, 1]) + min(pred[:, 1])) / 2, (max(pred[:, 2]) + min(pred[:, 2])) / 2],
                       zoom_start=5)

        for i in range(len(pred) - 1):
            folium.PolyLine([pred[i][1:], pred[i + 1][1:]], color="blue", weight=2.5, opacity=1).add_to(m)

        for i in range(len(true) - 1):
            folium.PolyLine([true[i][1:], true[i + 1][1:]], color="red", weight=2.5, opacity=1).add_to(m)

        if save_path:
            save_folder = os.path.dirname(save_path)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            m.save(save_path)
        else:
            m.save("path_comparison_map.html")

    def draw_path_img(pred, true, save_path=None):
        print("PAINTING PATH IMAGE")
        point_size = 5
        fig_size = (8, 6)

        plt.figure(figsize=fig_size)
        m = Basemap(projection='cyl', resolution='i', area_thresh=5000.)

        # draw coastline
        maptxt = coastline()
        for index, row in maptxt.iterrows():
            m.plot(row['m_lon'], row['m_lat'], latlon=True, color='black', linewidth=0.5)

        # draw predict line
        x_pred, y_pred = m(pred[:, 1], pred[:, 2]) # lon, lat
        m.plot(x_pred, y_pred, linestyle='-', marker='o', markersize=point_size, color='blue', label='_nolegend_')

        # draw ground truth
        x_true, y_true = m(true[:, 1], true[:, 2]) # lon, lat
        m.plot(x_true, y_true, linestyle='-', marker='o', markersize=point_size, color='red', label='_nolegend_')

        plt.legend()
        plt.title('Comparison between Predicted and True Paths')

        if save_path:
            save_folder = os.path.dirname(save_path)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            plt.savefig(save_path)
        else:
            plt.show()

    def calculate_similarity(pred, truth, method="mse"):
        n = len(pred)
        m = len(truth)
        Dis = tf.zeros((n, m), dtype=tf.float32)
        Acc = tf.zeros((n, m), dtype=tf.float32)

        for i in range(n):
            for j in range(m):
                if method == "mse":
                    Dis = tf.tensor_scatter_nd_update(Dis, [[i, j]], [weighted_mse(pred[i], truth[j])])
                elif method == "mae":
                    Dis = tf.tensor_scatter_nd_update(Dis, [[i, j]], [weighted_mae(pred[i], truth[j])])

        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    Acc = tf.tensor_scatter_nd_update(Acc, [[i, j]], [Dis[i, j]])
                elif i == 0 and j > 0:
                    Acc = tf.tensor_scatter_nd_update(Acc, [[i, j]], [Acc[i, j - 1] + Dis[i, j]])
                elif i > 0 and j == 0:
                    Acc = tf.tensor_scatter_nd_update(Acc, [[i, j]], [Acc[i - 1, j] + Dis[i, j]])
                else:
                    Acc = tf.tensor_scatter_nd_update(Acc, [[i, j]], [min(Acc[i - 1, j - 1], Acc[i - 1, j], Acc[i, j - 1]) + Dis[i, j]])

        d = tf.sqrt(1 - Acc[-1, -1])
        p = 1.005
        # Normalization
        numerator = p ** d - p ** (-d)
        denominator = p ** d + p ** (-d)
        return numerator / denominator

    for image_sequences, path_sequences, labels, feature, frame_ID_ascii, dInt, dLon, dLat in dataset:
        pred = model(image_sequences, path_sequences, feature, training=False)
        
        sample_weight = tf.math.tanh((dInt-20)/10)*1000 +1000.1
        # sample_weight = tf.nn.softmax(tf.math.tanh((dInt - 20) / 10) * 1000 + 1000.1)

        glo_mse_each = calculate_similarity(pred, labels, "mse")
        glo_mae_each = calculate_similarity(pred, labels, "mae")

        int_mae_each = tf.math.reduce_mean((tf.abs(labels[:, 0]-pred[:, 0]))*sample_weight)
        int_mse_each = tf.math.reduce_mean((tf.abs(labels[:, 0]-pred[:, 0])**2)*sample_weight)

        pred_lon_lat = pred[:, 1:]
        truth_lon_lat = labels[:, 1:]
        tck_mae_each = weighted_mae(truth_lon_lat, pred_lon_lat)
        tck_mse_each = weighted_mse(truth_lon_lat, pred_lon_lat)

        # path compare
        if draw_path == "IMG":
            timestamp = int(time.time())
            save_path = f"debug_helper/path_compare/path_comparison_map_{timestamp}.png"
            pred_inverse_normalized = inverse_normalize(pred, decode_frame_ID(frame_ID_ascii))
            labels_inverse_normalized = inverse_normalize(labels, decode_frame_ID(frame_ID_ascii))
            draw_path_img(pred_inverse_normalized, labels_inverse_normalized, save_path)
        elif draw_path == "MAP":
            timestamp = int(time.time())
            save_path = f"debug_helper/path_compare/path_comparison_map_{timestamp}.html"
            pred_inverse_normalized = inverse_normalize(pred, decode_frame_ID(frame_ID_ascii))
            # print("pred path", pred_inverse_normalized)
            labels_inverse_normalized = inverse_normalize(labels, decode_frame_ID(frame_ID_ascii))
            # print("true path", labels_inverse_normalized)
            draw_path_map(pred_inverse_normalized, labels_inverse_normalized, save_path)
        
        num+=1
        int_mae = tf.add(int_mae, int_mae_each)
        int_mse = tf.add(int_mse, int_mse_each)

        tck_mae = tf.add(tck_mae, tck_mae_each)
        tck_mse = tf.add(tck_mse, tck_mse_each)

        glo_mae = tf.add(glo_mae, glo_mae_each)
        glo_mse = tf.add(glo_mse, glo_mse_each)
        
    
    INTMAE = tf.reduce_mean(tf.math.divide(int_mae, num))
    INTMSE = tf.reduce_mean(tf.math.divide(int_mse, num))
    TCKMAE = tf.reduce_mean(tf.math.divide(tck_mae, num))
    TCKMSE = tf.reduce_mean(tf.math.divide(tck_mse, num))
    GLOMAE = tf.reduce_mean(tf.math.divide(glo_mae, num))
    GLOMSE = tf.reduce_mean(tf.math.divide(glo_mse, num))

    return dict(
        INTMAE=INTMAE,
        INTMSE=INTMSE,
        TCKMAE=TCKMAE,
        TCKMSE=TCKMSE,
        GLOMAE=GLOMAE,
        GLOMSE=GLOMSE
    )
