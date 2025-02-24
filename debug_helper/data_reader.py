'''
    Print the data structure of TCSA.h5
'''
from io import StringIO

import h5py
import pandas as pd
import os
import numpy as np

def write_block2_values_to_txt(h5_file_path, output_file):
    # 打开 HDF5 文件
    with h5py.File(h5_file_path, "r") as f:
        # 获取数据集
        dataset = f["/info/block2_values"]

        # 获取数据集的值
        data = dataset[:]

        # 打印数据的长度
        with open(output_file, 'w') as txt_file:
            txt_file.write("Length of data: {}\n".format(len(data[0])))

            # 打印前 80 个元素，每行 16 个数字
            for i in range(0, min(len(data[0]), 80), 16):
                row = ' '.join(map(str, data[0][i:i+16]))
                txt_file.write(row + '\n')

def print_h5_structure_to_txt(file_path, output_file):
    '''
    Explore the structure of H5 file.
    查看 H5 数据集的基本结构
    :param file_path:
    :param output_file:
    :return:
    '''
    with open(output_file, 'w') as txt_file:
        with h5py.File(file_path, 'r') as f:
            txt_file.write("文件中的对象列表: " + str(list(f.keys())) + "\n")
            stack = [("", f)]
            while stack:
                path, current = stack.pop()
                txt_file.write(" " * len(path) + path + "/" + current.name.split("/")[-1] + "\n")
                if isinstance(current, h5py.Group):
                    for name, item in current.items():
                        stack.append((path + "/" + current.name.split("/")[-1], item))
                elif isinstance(current, h5py.Dataset):
                    print_dataset_info_to_txt(current, txt_file, len(path) + 2)

def print_dataset_info_to_txt(dataset, txt_file, indent):
    txt_file.write(" " * indent + "Dataset: " + dataset.name + "\n")
    txt_file.write(" " * (indent + 2) + "Shape: " + str(dataset.shape) + "\n")
    txt_file.write(" " * (indent + 2) + "Data Type: " + str(dataset.dtype) + "\n")
    txt_file.write(" " * (indent + 2) + "First 24 elements:\n")
    for i in range(min(24, dataset.shape[0])):
        txt_file.write(" " * (indent + 4) + str(dataset[i]) + "\n")

def read_info_and_top_10_to_file(file_path, output_file):
    '''
    Get mini-dataset for debugging.
    :param file_path:
    :param output_file:
    :return:
    '''
    info_df = pd.read_hdf(file_path, key='info', mode='r')

    info_output = StringIO()
    info_df.info(buf=info_output)
    info_str = info_output.getvalue()
    top_10 = info_df.head(200)

    with open(output_file, 'w') as f:
        f.write("Data Structure Information:\n")
        f.write(info_str)
        f.write("\n\nTop 200 Data:\n")
        f.write(top_10.to_string())

def extract_h5_subset(input_file, output_file, num_samples=200):
    """
    Extract a subset from the HDF5 file while preserving the original structure.

    :param input_file: Path to the original .h5 file
    :param output_file: Path to save the extracted dataset
    :param num_samples: Number of samples to extract (default: 200)
    """
    with h5py.File(input_file, 'r') as infile, h5py.File(output_file, 'w') as outfile:
        # 复制 matrix
        if 'matrix' in infile:
            matrix_data = infile['matrix'][:num_samples]
            outfile.create_dataset('matrix', data=matrix_data)
        else:
            print("Warning: 'matrix' dataset not found in input file.")

    # 处理 info：读取 DataFrame -> 取前 num_samples 行 -> 存回 HDF5
    try:
        info_df = pd.read_hdf(input_file, key='info', mode='r')
        info_df_subset = info_df.iloc[:num_samples]  # 取前 num_samples 行
        info_df_subset.to_hdf(output_file, key='info', mode='a')  # 追加模式存回
    except Exception as e:
        print(f"Error processing 'info': {e}")

    print(f"✅ Subset extracted and saved to {output_file}!")



if __name__ == "__main__":
    # Extract debugging dataset
    h5_file_path = "../data/TCIR-CPAC_IO_SH/TCSA.h5"
    output_file_path = "../data/MINI-TCSA.h5"
    extract_h5_subset(h5_file_path, output_file_path)

    # Check mini-dataset structure
    mini_h5_file_path = "../data/MINI-TCSA.h5"
    structure_txt_path = "../debug_helper/debug_files/mini_info_sample.txt"
    # print_h5_structure_to_txt(mini_h5_file_path, structure_txt_path)
    read_info_and_top_10_to_file(mini_h5_file_path, structure_txt_path)

    # print("Structure of", h5_file_path)
    # print_h5_structure_to_txt(h5_file_path, output_file_path)
    # print("Data has been written to", output_file_path)