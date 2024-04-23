import os
import argparse
import arrow
from modules.experiment_helper import seed_everything, set_up_tensorflow, \
    parse_experiment_settings, get_model_save_path, get_summary_writer
from modules.model_constructor import create_model_instance
from modules.data_handler import get_tensorflow_datasets
from modules.model_trainer import train


def main(experiment_path, GPU_limit):
    seed_everything(seed=1126)
    set_up_tensorflow(GPU_limit)

    experiment_settings = parse_experiment_settings(experiment_path)
    experiment_name = experiment_settings['experiment_name']
    time_tag = arrow.now().format('YYYYMMDDHHmm')
    summary_writer = get_summary_writer(experiment_name, time_tag)
    model_save_path = get_model_save_path(experiment_name) # model will be saved at save_models/{experiment_name}

    datasets = get_tensorflow_datasets(**experiment_settings['data'])

    # Print single sample in the dataset.
    # print("Structure of a single sample in 'train' dataset:")
    # print(datasets['train'].element_spec)

    # # phase sample in dataset['train'] one by one.
    # for sample in datasets['train']:
    #     # 解析每个样本中的元素
    #     image_sequences, path_sequences, labels, feature, frame_ID_ascii, dInt, dLon, dLat = sample
    #     # 进行您的操作，例如打印每个元素的形状等
    #     print("Image sequences shape:", image_sequences.shape)
    #     print("Path sequences shape:", path_sequences.shape)
    #     print("Labels shape:", labels.shape)
    #     print("Feature shape:", feature.shape)
    #     print("Frame ID ASCII shape:", frame_ID_ascii.shape)
    #     print("dInt shape:", dInt.shape)
    #     print("dLon shape:", dLon.shape)
    #     print("dLat shape:", dLat.shape)

    model = create_model_instance(experiment_settings['model'])

    train(
        model,
        datasets,
        summary_writer,
        model_save_path,
        **experiment_settings['training_setting']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", help="path of config file")
    parser.add_argument('--GPU_limit', type=int, default=8000)
    parser.add_argument('-d', '--CUDA_VISIBLE_DEVICES', type=str, default='')
    args = parser.parse_args()

    if args.CUDA_VISIBLE_DEVICES:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    main(args.experiment_path, args.GPU_limit)
