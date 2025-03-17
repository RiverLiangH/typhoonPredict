import os
import argparse
import tensorflow as tf
from modules.experiment_helper import seed_everything, set_up_tensorflow, parse_experiment_settings
from modules.model_constructor import create_model_instance
from modules.data_handler import get_tensorflow_datasets
import numpy as np

from modules.training_helper import inverse_normalize, decode_frame_ID


def load_model(checkpoint_path, model_config):
    """ 加载模型权重 """
    model = create_model_instance(model_config)  # 重新创建模型结构
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    return model

def evaluate_model(model, dataset):
    """ 计算模型预测值和实际值的对比 """
    predictions, ground_truths = [], []

    for image_sequences, labels, feature, frame_ID_ascii, dInt in dataset:
        raw_preds = model(image_sequences, feature, training=False)  # 进行预测

        # 反归一化
        preds = inverse_normalize(raw_preds, decode_frame_ID(frame_ID_ascii))
        labels = inverse_normalize(labels, decode_frame_ID(frame_ID_ascii))

        labels = tf.cast(labels, tf.float32)
        preds = tf.cast(preds, tf.float32)

        predictions.extend(preds.numpy())  # 转 numpy 数组
        ground_truths.extend(labels.numpy())  # 转 numpy 数组

    return predictions, ground_truths if labels is not None else None


def main(experiment_path, GPU_limit):
    seed_everything(seed=1126)
    set_up_tensorflow(GPU_limit)

    # 解析实验配置
    experiment_settings = parse_experiment_settings(experiment_path)
    experiment_name = experiment_settings['experiment_name']

    # 获取数据集
    datasets = get_tensorflow_datasets(**experiment_settings['data'])
    test_dataset = datasets['test']  # 假设 'test' 为测试集

    # 加载模型
    checkpoint_path = f"save_models/{experiment_name}/"
    model = load_model(checkpoint_path, experiment_settings['model'])

    # 进行预测
    preds, actuals = evaluate_model(model, test_dataset)

    # 打印对比结果
    for i in range(10):  # 只打印前10个对比
        print(f"Actual: {actuals[i]}, Predicted: {preds[i]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", help="path of config file")
    parser.add_argument('--GPU_limit', type=int, default=8000)
    parser.add_argument('-d', '--CUDA_VISIBLE_DEVICES', type=str, default='')
    args = parser.parse_args()

    if args.CUDA_VISIBLE_DEVICES:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES  # 修正 typo

    main(args.experiment_path, args.GPU_limit)