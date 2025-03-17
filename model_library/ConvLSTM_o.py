import tensorflow as tf
from tensorflow.keras import layers

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # 输入标准化
        self.input_norm = layers.BatchNormalization()

        # 图像编码部分
        self.image_encoder_layers = [
            layers.Conv2D(filters=16, kernel_size=4, strides=2, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
            layers.BatchNormalization()
        ]

        # RNN部分（如果不需要复杂的时序特征，考虑改为Dense层）
        self.rnn_block = layers.ConvLSTM2D(
            filters=64, kernel_size=4, dropout=0.0,
            recurrent_dropout=0.0, return_sequences=False
        )

        # RNN输出编码部分
        self.rnn_output_encoder = layers.Conv2D(filters=64, kernel_size=1, strides=1, activation='relu')

        # 输出层
        self.output_layers = [
            layers.Dense(units=128, activation='relu'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=1),  # 线性输出
        ]

    def apply_list_of_layers(self, input, list_of_layers, training):
        x = input
        for layer in list_of_layers:
            x = layer(x, training=training)
        return x

    def auxiliary_feature(self, feature):
        # 特征: ['starting_land_dis', 'ending_land_dis', 'translation_speed', 'starting_intensity', 'starting_lat', 'ending_lat']
        land_distance = feature[:, 0:2]
        translation_speed = feature[:, 2:6]
        return tf.concat([land_distance, translation_speed], 1)

    def call(self, image_sequences, feature, training):
        batch_size, encode_length, height, width, channels = image_sequences.shape

        # 图像编码块
        images = tf.reshape(
            image_sequences, [batch_size * encode_length, height, width, channels]
        )
        normalized_images = self.input_norm(images, training=training)
        encoded_images = self.apply_list_of_layers(
            normalized_images, self.image_encoder_layers, training
        )
        total_image_counts, height, width, channels = encoded_images.shape
        encoded_image_sequences = tf.reshape(
            encoded_images, [batch_size, encode_length, height, width, channels]
        )

        # RNN块
        feature_sequences = self.rnn_block(encoded_image_sequences, training=training)

        # RNN输出编码
        compressed_features = self.rnn_output_encoder(
            feature_sequences, training=training
        )
        flatten_feature = tf.reshape(compressed_features, [batch_size, -1])

        # 启用辅助特征
        auxiliary_feature = self.auxiliary_feature(feature)

        # 拼接图像特征和辅助特征
        combine_feature = tf.concat([flatten_feature, auxiliary_feature], 1)

        # 输出层
        output = self.apply_list_of_layers(
            combine_feature, self.output_layers, training
        )
        return output
