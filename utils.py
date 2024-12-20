# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:18:58 2024

@author: 18307
"""
import os
import pandas
import numpy
import h5py
import scipy

# 定义一个通用的归一化函数
def safe_normalize(data):
    """
    对输入数据逐样本归一化，适用于 2D 和 3D 数据。
    
    Parameters:
        data (np.ndarray): 输入数据，形状为 (samples, n, m) 或 (samples, n, m, l)。
    
    Returns:
        np.ndarray: 归一化后的数据。
    """
    # 检查输入的维度
    if len(data.shape) == 3:  # 2D 数据
        axis = (1, 2)
    elif len(data.shape) == 4:  # 3D 数据
        axis = (1, 2, 3)
    else:
        raise ValueError("Input data must have shape (samples, n, m) or (samples, n, m, l)")
    
    # 计算每个样本的最大值
    max_values = numpy.max(data, axis=axis, keepdims=True)
    
    # 避免除以零
    epsilon = 1e-8
    max_values[max_values == 0] = epsilon
    
    # 逐样本归一化
    return data / max_values

def get_label():
    # path
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    
    path_labels = os.path.join(path_parent, 'data', 'SEED', 'functional connectivity', 'labels.txt')
    
    # read txt; original channel distribution
    labels = pandas.read_csv(path_labels, sep='\t', header=None).to_numpy().flatten()
    
    print('Labels Reading Done')
    
    return labels

def read_labels(path_txt):
    # read txt; original channel distribution
    labels = pandas.read_csv(path_txt, sep='\t', header=None).to_numpy().flatten()
    
    print('Labels Reading Done')
    
    return labels

def get_distribution(mapping_method):
    # define path
    path_current = os.getcwd()
    
    # read distribution txt
    if mapping_method == 'auto':
        path_ch_auto_distr = os.path.join(path_current, 
                                          'channel_distribution', 
                                          'biosemi62_64_channels_original_distribution.txt')
        # read txt; channel distribution
        distribution = pandas.read_csv(path_ch_auto_distr, sep='\t')
        
    elif mapping_method == 'manual':
        path_ch_manual_distr = os.path.join(path_current, 
                                            'channel_distribution', 
                                            'biosemi62_64_channels_manual_distribution.txt')    
        # read txt; channel distribution
        distribution = pandas.read_csv(path_ch_manual_distr, sep='\t')
    
    return distribution

def read_distribution(path_txt):
    # read txt; channel distribution
    distribution = pandas.read_csv(path_txt, sep='\t')
    
    return distribution

def get_channel_feature_mat(feature, band, experiment):
    # path
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    
    # path_data
    path_data = os.path.join(path_parent, 'data', 'SEED', 'channel features', feature, experiment + '.mat')
    
    # mat data
    mat_data = read_mat(path_data)
    
    # determine specific band
    first_key = next(iter(mat_data))
    match band:
        case 'delta':
            mat_data = mat_data.get(first_key)[0]
        case 'theta':
            mat_data = mat_data.get(first_key)[1]
        case 'alpha':
            mat_data = mat_data.get(first_key)[2]
        case 'beta':
            mat_data = mat_data.get(first_key)[3]
        case 'gamma':
            mat_data = mat_data.get(first_key)[4]
        case _:
            raise TypeError('frequency band not match!\n')

    return mat_data

def read_mat(path_file):
    # 确保文件存在
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        # 尝试以 HDF5 格式读取文件
        with h5py.File(path_file, 'r') as f:
            print("HDF5 format detected.")
            # 提取所有键值及其数据
            mat_data = {key: numpy.array(f[key]) for key in f.keys()}
    
    except OSError:
        # 如果不是 HDF5 格式，尝试使用 scipy.io.loadmat
        print("Not an HDF5 format.")
        mat_data = scipy.io.loadmat(path_file)
        # 排除系统默认的键
        mat_data = {key: mat_data[key] for key in mat_data.keys() if not key.startswith('__')}
    
    return mat_data

def cmdata_reshaper(mat_data):
    """
    Reshapes mat_data to ensure the last two dimensions are square (n1 == n2).
    Automatically handles transposing and validates the shape.
    """
    MAX_ITER = 10  # 最大迭代次数，防止死循环
    iteration = 0

    while iteration < MAX_ITER:
        if mat_data.ndim == 3:
            samples, n1, n2 = mat_data.shape
            if n1 == n2:
                break  # 如果满足条件，直接退出
            else:
                mat_data = numpy.transpose(mat_data, axes=(2, 0, 1))  # 转置调整维度
        iteration += 1

    else:
        raise ValueError("Failed to reshape mat_data into (samples, n1, n2) with n1 == n2 after multiple attempts.")

    return mat_data

def read_mat_single_idx_3chs(path_data, name_feature, name_experiment):
    path_file = os.path.join(path_data, name_feature, name_experiment)
    with h5py.File(path_file, 'r') as mat_file:
        # 查看文件中的所有键
        # print("Keys in the .mat file:", list(mat_file.keys()))
        
        # 假设.mat文件中包含一个数据集 'data'
        # data = mat_file['de_LDS']
        data = mat_file[name_feature]
        
        # 将数据转换为numpy数组并打印
        data_array = data[:]
        data_alpha = data_array[2]
        data_beta = data_array[3]
        data_gamma = data_array[4]
        
        return data_alpha, data_beta, data_gamma

def read_mat_single_idx_5chs(path_data, name_feature, name_experiment):
    path_file = os.path.join(path_data, name_feature, name_experiment)
    with h5py.File(path_file, 'r') as mat_file:
        # 查看文件中的所有键
        # print("Keys in the .mat file:", list(mat_file.keys()))
        
        # 假设.mat文件中包含一个数据集 'data'
        # data = mat_file['de_LDS']
        data = mat_file[name_feature]
        
        # 将数据转换为numpy数组并打印
        data_array = data[:]
        data_delta = data_array[0]
        data_theta = data_array[1]
        data_alpha = data_array[2]
        data_beta = data_array[3]
        data_gamma = data_array[4]
        
        return data_delta, data_theta, data_alpha, data_beta, data_gamma
    
# # test code
# path_current = os.getcwd()
# path_parent = os.path.dirname(path_current)
# path_data = os.path.join(path_parent, 'data', 'SEED', 'channel features')

# # data_alpha, data_beta, data_gamma = read_mat_single_idx_3chs(path_data, 'de_LDS', 'sub1ex1.mat')
# data_alpha1 = read_mat(os.path.join(path_data, 'de_LDS', 'sub1ex1.mat'))
# data_alpha2 = get_channel_feature_mat('de_LDS', 'alpha', 'sub1ex1')