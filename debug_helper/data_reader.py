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

if __name__ == "__main__":
    h5_file_path = "../TCSA_data/debug.h5"
    output_file_path = "debug_files/origin_debug.txt"
    # write_block2_values_to_txt(h5_file_path, output_file_path)

    read_info_and_top_10_to_file(h5_file_path, output_file_path)
    # print("Structure of", h5_file_path)
    # print_h5_structure_to_txt(h5_file_path, output_file_path)
    # print("Data has been written to", output_file_path)