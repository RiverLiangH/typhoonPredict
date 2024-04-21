import tensorflow as tf
import time
from modules.tfrecord_generator import *
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def calculate_metric_dict(model, dataset):
    mae = tf.constant([0.])
    mse = tf.constant([0.])
    num = 0.

    def draw_path(pred, true, save_path=None):
        print("PAINTING PATH IMAGE")
        m = Basemap(projection='cyl', resolution='i', area_thresh=5000.)

        # draw coastline
        maptxt = coastline()
        for index, row in maptxt.iterrows():
            m.plot(row['m_lon'], row['m_lat'], latlon=True, color='black', linewidth=0.5)

        # draw predict line
        x_pred, y_pred = m(pred[:, 1], pred[:, 2]) # lon, lat
        m.scatter(x_pred, y_pred, marker='o', color='blue', label='Predicted Path')

        # draw ground truth
        x_true, y_true = m(true[:, 1], true[:, 2]) # lon, lat
        m.scatter(x_true, y_true, marker='o', color='red', label='True Path')

        plt.legend()
        plt.title('Comparison between Predicted and True Paths')

        if save_path:
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
                    Dis[i, j] = weighted_mse(pred[i], truth[j])
                elif method == "mae":
                    Dis[i, j] = weighted_mae(pred[i], truth[j])

        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    Acc[i, j] = Dis[i, j]
                elif i == 0 and j > 0:
                    Acc[i, j] = Acc[i, j - 1] + Dis[i, j]
                elif i > 0 and j == 0:
                    Acc[i, j] = Acc[i - 1, j] + Dis[i, j]
                else:
                    Acc[i, j] = min(Acc[i - 1, j - 1], Acc[i - 1, j], Acc[i, j - 1]) + Dis[i, j]

        d = tf.sqrt(Acc)
        p = 1.005
        # Normalization
        numerator = p ** d - p ** (-d)
        denominator = p ** d + p ** (-d)
        return numerator / denominator

    for image_sequences, path_sequences, labels, feature, frame_ID_ascii, dInt, dLon, dLat in dataset:
        pred = model(image_sequences, path_sequences, feature, training=False)
        
        # sample_weight = tf.math.tanh((dInt-20)/10)*1000 +1000.1
        # sample_weight = tf.nn.softmax(tf.math.tanh((dInt - 20) / 10) * 1000 + 1000.1)

        mse_each = calculate_similarity(pred, labels, "mse")
        mae_each = calculate_similarity(pred, labels, "mae")

        # mae_each = tf.math.reduce_mean((tf.abs(labels[:, 0]-pred[:, 0]))*sample_weight)
        # mse_each = tf.math.reduce_mean((tf.abs(labels[:, 0]-pred[:, 0])**2)*sample_weight)
        # path compare
        timestamp = int(time.time())
        save_path = f"../debug_helper/path_compare/path_comparison_map_{timestamp}.png"
        draw_path(pred, labels, save_path)
        
        num+=1
        mae = tf.add(mae, mae_each)
        mse = tf.add(mse, mse_each)
        
    
    MAE = tf.reduce_mean(tf.math.divide(mae, num))
    MSE = tf.reduce_mean(tf.math.divide(mse, num))

    return dict(
        MAE=MAE,
        MSE=MSE
    )
