import tensorflow as tf
from modules.model_constructor import create_model_instance
from modules.experiment_helper import seed_everything, set_up_tensorflow, \
    parse_experiment_settings, get_model_save_path, get_summary_writer
from modules.data_handler import get_tensorflow_datasets

experiment_settings = parse_experiment_settings("experiments/ens21.yml")
datasets = get_tensorflow_datasets(**experiment_settings['data'])

model = create_model_instance("ConvLSTM_CCA_Pro")  # 替换为你的模型类的实例化
model.load_weights("saved_models/ens21/best-MAE")

for image_sequences, path_sequences, labels, feature, frame_ID_ascii, dInt, dLon, dLat in datasets['valid']:
    result = model(image_sequences, path_sequences, feature, training=False)  # 替换为你的预测代码
    # print("Path Sequences", path_sequences)
    # print("Image Sequences", image_sequences)
    # print("Feature:", feature)
    print("frame_ID_ascii:", frame_ID_ascii)
    print("Labels:", labels)
    print("Output values:", result.numpy())