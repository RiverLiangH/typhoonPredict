import tensorflow as tf
import time
import os
from modules.tfrecord_generator import *
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def calculate_metric_dict(model, dataset, draw_path_img = False):
    mae = tf.constant([0.])
    mse = tf.constant([0.])
    glo_mae = tf.constant([0.])
    glo_mse = tf.constant([0.])
    num = 0.

    def draw_path(pred, true, save_path=None):
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
        m.plot(x_pred, y_pred, s=point_size, marker='o', color='blue', label='_nolegend_')

        # draw ground truth
        x_true, y_true = m(true[:, 1], true[:, 2]) # lon, lat
        m.plot(x_true, y_true, s=point_size, marker='o', color='red', label='_nolegend_')

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
        def weighted_mse(p1, p2):
            squared_diff = tf.square(p1 - p2)
            mse = tf.reduce_sum(squared_diff)
            return mse

        def weighted_mae(p1, p2):
            abs_diff = tf.abs(p1 - p2)
            abs_diff_sum = tf.reduce_sum(abs_diff)
            return abs_diff_sum

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

        d = tf.sqrt(Acc[-1, -1])
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

        mae_each = tf.math.reduce_mean((tf.abs(labels[:, 0]-pred[:, 0]))*sample_weight)
        mse_each = tf.math.reduce_mean((tf.abs(labels[:, 0]-pred[:, 0])**2)*sample_weight)

        # path compare
        if draw_path_img:
            timestamp = int(time.time())
            save_path = f"debug_helper/path_compare/path_comparison_map_{timestamp}.png"
            draw_path(pred, labels, save_path)
        
        num+=1
        mae = tf.add(mae, mae_each)
        mse = tf.add(mse, mse_each)
        glo_mae = tf.add(glo_mae, glo_mae_each)
        glo_mse = tf.add(glo_mse, glo_mse_each)
        
    
    MAE = tf.reduce_mean(tf.math.divide(mae, num))
    MSE = tf.reduce_mean(tf.math.divide(mse, num))
    GLOMAE = tf.reduce_mean(tf.math.divide(glo_mae, num))
    GLOMSE = tf.reduce_mean(tf.math.divide(glo_mse, num))

    return dict(
        MAE=MAE,
        MSE=MSE,
        GLOMAE=GLOMAE,
        GLOMSE=GLOMSE
    )
