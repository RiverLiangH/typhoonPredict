'''
    data preprocessing
'''

import tensorflow as tf
from functools import partial
import tensorflow_addons as tfa
from modules.tfrecord_generator import get_or_generate_tfrecord


def ascii_array_to_string(ascii_array):
    string = ''
    for ascii_code in ascii_array:
        string += chr(ascii_code)
    return string

def deserialize(serialized_TC_history):
    '''
    < Global Training Dataset Structure >
        - Mainly call tf.io.parse_single_example to cope with the data.
        - returns a dictionary (or similar data structure) containing the features extracted from the serialized example.
        - Each feature is represented as a TensorFlow tensor. (Similar to JSON)
    :param serialized_TC_history:
    :return:
    '''
    features = {
        'history_len': tf.io.FixedLenFeature([], tf.int64),
        'images': tf.io.FixedLenFeature([], tf.string),
        'intensity': tf.io.FixedLenFeature([], tf.string),
        'frame_ID': tf.io.FixedLenFeature([], tf.string),
        'lon': tf.io.FixedLenFeature([], tf.string),
        'lat': tf.io.FixedLenFeature([], tf.string),
        'env_feature': tf.io.FixedLenFeature([], tf.string),
        'SHTD': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(serialized_TC_history, features)
    history_len = tf.cast(example['history_len'], tf.int32) # suppose 'history_len' indicates time series.

    images = tf.reshape(
        tf.io.decode_raw(example['images'], tf.float32),
        [history_len, 64, 64, 4]
    )
    intensity = tf.reshape(
        tf.io.decode_raw(example['intensity'], tf.float64),
        [history_len]
    )
    intensity = tf.cast(intensity, tf.float32)
    
    lon = tf.reshape(
        tf.io.decode_raw(example['lon'], tf.float64),
        [history_len]
    )    
    lon = tf.cast(lon, tf.float32)
    
    lat = tf.reshape(
        tf.io.decode_raw(example['lat'], tf.float64),
        [history_len]
    )    
    lat = tf.cast(lat, tf.float32)

    env_feature = tf.reshape(
        tf.io.decode_raw(example['env_feature'], tf.float64),
        [-1 ,history_len]
    )    
    env_feature = tf.cast(env_feature, tf.float32)
    
    SHTD = tf.reshape(
        tf.io.decode_raw(example['SHTD'], tf.float64),
        [history_len]
    )    
    SHTD = tf.cast(SHTD, tf.float32)

    frame_ID_ascii = tf.reshape(
        tf.io.decode_raw(example['frame_ID'], tf.uint8),
        [history_len, -1]
    )

    return images, intensity, lon, lat, env_feature, history_len, frame_ID_ascii, SHTD



def translation(starting_lon, starting_lat, ending_lon, ending_lat, estimate_distance):
    '''
    Calculate the moving speed of typhoon.
    :param starting_lon:
    :param starting_lat:
    :param ending_lon:
    :param ending_lat:
    :param estimate_distance:
    :return:
    '''
    west = 360. * tf.cast((ending_lon < 0), tf.float32)
    ending_lon = tf.cast((ending_lon), tf.float32) + west

    west = 360. * tf.cast((starting_lon < 0), tf.float32)
    starting_lon = tf.cast((starting_lon), tf.float32) + west

    lon_dif = tf.math.abs((ending_lon-starting_lon))
    cross_0 = 360. * tf.cast((lon_dif > 100), tf.float32)
    lon_dif = cross_0 - lon_dif
    
    lat_dif= tf.cast(ending_lat-starting_lat, tf.float32)
    
    speed = 110*tf.sqrt(tf.square(lon_dif) + tf.square(lat_dif))/(estimate_distance*3)    # in  km/hr
    return speed

def breakdown_into_sequence(
    images, intensity, lon, lat, env_feature, history_len, frame_ID_ascii, encode_length, estimate_distance
):
    '''

    :param images:
    :param intensity:
    :param lon:
    :param lat:
    :param env_feature:
    :param history_len: represents the length of the historical sequence.
    :param frame_ID_ascii:
    :param encode_length: previous time step used for prediction.
    :param estimate_distance: time step of prediction.
    :return:
    '''
    sequence_num = history_len - (encode_length + estimate_distance) + 1
    starting_index = tf.range(2, sequence_num) # tensor: [2, sequence_num-1]

    image_sequences = tf.map_fn(
        lambda start: images[start: start+encode_length],
        starting_index, fn_output_signature=tf.float32
    )

    path_data = tf.stack([lon, lat], axis=-1)
    path_sequences = tf.map_fn(
        lambda start: path_data[start: start + encode_length],
        starting_index,
        fn_output_signature=tf.float32
    )

    starting_frame_ID_ascii = frame_ID_ascii[encode_length + 1:-estimate_distance]

    # intensity_change
    previous_6hr_intensity = intensity[encode_length - 1: -estimate_distance -2]
    starting_intensity = intensity[encode_length + 1: -estimate_distance]
    ending_intensity = intensity[encode_length + estimate_distance + 1:]
    intensity_change = ending_intensity - starting_intensity

    starting_lon = lon[encode_length + 1: -estimate_distance]
    ending_lon = lon[encode_length + estimate_distance + 1:]
    lon_change = ending_lon - starting_lon

    starting_lat = lat[encode_length + 1: -estimate_distance]
    ending_lat = lat[encode_length + estimate_distance + 1:]
    lat_change = ending_lat - starting_lat

    labels = tf.concat([ending_intensity, ending_lat, ending_lon], axis=-1)

    # starting_lon = lon[encode_length + 1: -estimate_distance]
    # ending_lon = lon[encode_length + estimate_distance + 1:]
    # starting_lat = lat[encode_length + 1: -estimate_distance]
    # ending_lat = lat[encode_length + estimate_distance + 1:]
   
    translation_speed = translation(starting_lon, starting_lat, ending_lon, ending_lat, estimate_distance)

    starting_lat = tf.math.abs(starting_lat)
    ending_lat = tf.math.abs(ending_lat)  
    
    starting_env_feature = env_feature[:, encode_length + 1:-estimate_distance]
    ending_env_feature = env_feature[:, encode_length + estimate_distance + 1:]
    
    feature = tf.concat([[starting_lat], [ending_lat], [translation_speed], [starting_intensity], [previous_6hr_intensity], starting_env_feature, ending_env_feature], 0)
    feature = tf.transpose(feature)

    return tf.data.Dataset.from_tensor_slices((image_sequences, path_sequences, labels, feature, starting_frame_ID_ascii, intensity_change, lon_change, lat_change))


def image_preprocessing(images, intensity, lon, lat, env_feature, history_len, frame_ID_ascii, SHTD, rotate_type, input_image_type):
    '''
    Operating at every typhoon record (in the format of time series).
    func:
        - Filtering colors. Only specific color (set at .yml files) will be left.
        - Operating rotation.
    :param images: [history_len, 64, 64, 4]
    :param intensity:
    :param lon:
    :param lat:
    :param env_feature:
    :param history_len:
    :param frame_ID_ascii:
    :param SHTD:
    :param rotate_type:      configure at ensXX.yml file, indicating the way of rotating operation.
    :param input_image_type: configure at ensXX.yml file
    :return:
    '''
    images_channels = tf.gather(images, input_image_type, axis=-1) # Only certain color channels are of interest, while others are ignored.
    '''
    tf.gather
        tf.gather: This is a TensorFlow function used for gathering slices from a tensor 
                    along a specified axis according to indices provided.
        '-1': the last axis of the images tensor.
        return: Tensor.
    '''
    if rotate_type == 'single':
        angles = tf.random.uniform([history_len], maxval=360)
        rotated_images = tfa.image.rotate(images_channels, angles=angles)
        # A series of random angles is generated, and images_channels is rotated using these angles.
    elif rotate_type == 'series':
        angles = tf.ones([history_len]) * tf.random.uniform([1], maxval=360)
        rotated_images = tfa.image.rotate(images_channels, angles=angles)
        # A single random angle is generated and applied to all images.
    elif rotate_type == 'shear':
        print('this is the shear rotation run')
        rotated_images = tfa.image.rotate(images_channels, angles=-SHTD*0.01745329252)
        # a shear rotation is applied. The images_channels is rotated using the shear angle specified.
    else:
        rotated_images = images_channels # without any rotation.
    return rotated_images, intensity, lon, lat, env_feature, history_len, frame_ID_ascii


def get_tensorflow_datasets(
    data_folder, batch_size, encode_length,
    estimate_distance, rotate_type,input_image_type
):
    '''
    get_tensorflow_datasets
        process and transform serialized typhoon data stored in TFRecord files,
        generating TensorFlow datasets suitable for training or testing purposes.
        Raw Dataset --> TFRecord file --> tf.data.TFRecordDataset
    :param data_folder:
    :param batch_size:
    :param encode_length:
    :param estimate_distance: time step of prediction.
    :param rotate_type:
    :param input_image_type:
    :return:
    '''
    tfrecord_paths = get_or_generate_tfrecord(data_folder)
    print("data_folder", data_folder)
    datasets = dict()
    for phase, record_path in tfrecord_paths.items():
        '''
            'tf.data.TFRecordDataset' is used in TensorFlow to create a dataset from one or more TFRecord files. 
            TFRecord files are a binary format used to efficiently store large datasets. 
            The TFRecordDataset class allows you to read data from TFRecord files and convert it into a TensorFlow dataset, 
            which can then be used as input for training or evaluation of machine learning models.
        '''
        serialized_TC_histories = tf.data.TFRecordDataset(
            [record_path], num_parallel_reads=8
        ) # load data from TFRecord files path (record_path)
        TC_histories = serialized_TC_histories.map(
            deserialize, num_parallel_calls=tf.data.AUTOTUNE
        ) # for each ele ['train','valid','test'] in serialized_TC_histories (map) call deserialize to process.

        min_history_len = encode_length + estimate_distance + 2  # +2 for extra 6hr information
        long_enough_histories = TC_histories.filter(
            lambda a, b, c, d, e, f, g, h: f >= min_history_len
        ) # filter

        # image pre-process
        preprocessed_histories = long_enough_histories.map(
            partial(
                image_preprocessing,
                rotate_type=rotate_type,
                input_image_type=input_image_type
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
        TC_sequence = preprocessed_histories.interleave(
                partial(
                breakdown_into_sequence,
                encode_length=encode_length,
                estimate_distance=estimate_distance
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = TC_sequence.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(4)
        datasets[phase] = dataset

    return datasets
