import numpy as np
import tensorflow as tf
from collections import defaultdict
from modules.training_helper import calculate_metric_dict


def train(
    model,
    datasets,
    summary_writer,
    saving_path,
    max_epoch,
    evaluate_freq,
    class_weight,
    learning_rate
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_function = tf.keras.losses.MeanSquaredError()
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))
    class_weight, norm = tf.linalg.normalize(tf.cast(class_weight, tf.float32), ord=1)

    '''
      Operating one training step, calculate and update loss value.
    '''
    # self-defined loss function for intensity
    def intensity_loss(y_true, y_pred):
        return tf.keras.losses.MeanSquaredError()(y_true[:, 0], y_pred[:, 0])

    # self-defined loss function for path
    def location_loss(y_true, y_pred):
        lat_true, lon_true = y_true[:, 1], y_true[:, 2]
        lat_pred, lon_pred = y_pred[:, 1], y_pred[:, 2]
        return tf.sqrt(tf.pow(lat_pred - lat_true, 2) + tf.pow(lon_pred - lon_true, 2))

    # self-defined loss function for both intensity and path
    def total_loss_function(y_true, y_pred):
        intensity_loss_value = intensity_loss(y_true, y_pred)
        location_loss_value = location_loss(y_true, y_pred)
        return intensity_loss_value + location_loss_value

    # intensity-prediction
    @tf.function
    def train_step(model, image_sequences, labels, feature, dV):
        with tf.GradientTape() as tape:  # Creates a context manager GradientTape used for computing gradients.
            model_output = model(image_sequences, feature, training=True)  # iterate training.

            sample_weight = tf.math.tanh((dV - 20) / 10) * 1000 + 1000.1
            sample_weight = tf.expand_dims(sample_weight, axis=1)
            batch_loss = loss_function(labels, model_output, sample_weight=sample_weight)

        gradients = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        avg_losses['mean square error'].update_state(batch_loss)
        return

    # path-prediction
    @tf.function
    def train_step_path(model, image_sequences, labels, feature, dV):
        with tf.GradientTape() as tape:  # Creates a context manager GradientTape used for computing gradients.
            model_output = model(image_sequences, feature, training=True)  # iterate training.

            sample_weight = tf.math.tanh((dV - 20) / 10) * 1000 + 1000.1
            sample_weight = tf.expand_dims(sample_weight, axis=1)
            batch_loss = loss_function(labels, model_output, sample_weight=sample_weight)

        gradients = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        avg_losses['mean square error'].update_state(batch_loss)
        return

    # path&intensity prediction
    @tf.function
    def train_step_merge(model, image_sequences, path_sequences, labels, feature, dInt, dLon, dLat):
        with tf.GradientTape() as tape:  # Creates a context manager GradientTape used for computing gradients.
            model_output = model(image_sequences, path_sequences, feature, training=True)  # iterate training.

            # sample_weight = tf.math.tanh((dInt - 20) / 10) * 1000 + 1000.1
            # sample_weight = tf.expand_dims(sample_weight, axis=1)
            batch_loss = total_loss_function(labels, model_output)

        gradients = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        avg_losses['mean square error'].update_state(batch_loss)
        return

    # debug
    # for image_sequences, path_sequences, labels, feature, frame_ID_ascii, dInt, dLon, dLat in datasets['train']:
    #     output = model(image_sequences, path_sequences, feature, training=False)
    #     print("Output shape:", output.shape)
    #     print("Output values:", output.numpy())


    best_MAE = np.inf
    best_MSE = np.inf
    for epoch_index in range(1, max_epoch+1): # max_epoch: max training iteration.
        print(f'Executing epoch #{epoch_index}')

        for image_sequences, path_sequences, labels, feature, frame_ID_ascii, dInt, dLon, dLat in datasets['train']:
            train_step_merge(model, image_sequences, path_sequences, labels, feature,  dInt, dLon, dLat)

        # for image_sequences, labels, feature, frame_ID_ascii, dV  in datasets['train']:
        #     train_step(model, image_sequences, labels, feature, dV)

        with summary_writer['train'].as_default():
            for loss_name, avg_loss in avg_losses.items():
                tf.summary.scalar(loss_name, avg_loss.result(), step=epoch_index)
                avg_loss.reset_states()

        if epoch_index % evaluate_freq == 0:
            print(f'Completed {epoch_index} epochs, do some evaluation')

            for phase in [ 'test', 'valid']:
                if epoch_index >= 28:
                    metric_dict = calculate_metric_dict(model, datasets[phase], draw_path_img=True)
                else:
                    metric_dict = calculate_metric_dict(model, datasets[phase])
                with summary_writer[phase].as_default():
                    for metric_name, metric_value in metric_dict.items():
                        tf.summary.scalar(metric_name, metric_value, step=epoch_index)

            valid_MAE = metric_dict['MAE']
            valid_MSE = metric_dict['MSE']
            if best_MAE > valid_MAE:
                best_MAE = valid_MAE
                model.save_weights(saving_path/'best-MAE', save_format='tf')
            if best_MSE > valid_MSE:
                best_MSE = valid_MSE
                model.save_weights(saving_path/'best-MSE', save_format='tf')
