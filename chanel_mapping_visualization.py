# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:12:01 2024

@author: usouu
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import utils
import channel_mapping_2d
import channel_mapping_3d

def show_nonzeros_2d(matrix):
    x, y = np.indices(matrix.shape)
    
    non_zero_mask = matrix != 0
    x_non_zero = x[non_zero_mask]
    y_non_zero = y[non_zero_mask]
    colors_non_zero = matrix[non_zero_mask]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x_non_zero, y_non_zero, c=colors_non_zero, cmap='viridis')

    plt.colorbar(sc, ax=ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
def show_nonzeros_3d(matrix):
    x, y, z = np.indices(matrix.shape)
    
    non_zero_mask = matrix != 0
    x_non_zero = x[non_zero_mask]
    y_non_zero = y[non_zero_mask]
    z_non_zero = z[non_zero_mask]
    colors_non_zero = matrix[non_zero_mask]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x_non_zero, y_non_zero, z_non_zero, c=colors_non_zero, cmap='viridis')

    plt.colorbar(sc, ax=ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
# define path
path_current = os.getcwd()
path_parent = os.path.dirname(path_current)

# read data
path_data = os.path.join(path_parent, 'data', 'SEED', 'channel features')
data_alpha, data_beta, data_gamma = utils.read_mat_single_idx_3chs(path_data, 'de_LDS', 'sub1ex1.mat')

# read channel matrix
path_ch_distr_auto = os.path.join(path_current, 
                                  'channel_distribution', 
                                  'biosemi62_64_channels_original_distribution.txt')
path_ch_distr_manual = os.path.join(path_current, 
                                    'channel_distribution', 
                                    'biosemi62_64_channels_manual_distribution.txt')

distribution = utils.get_distribution(path_ch_distr_auto)

# 2d samples
# alpha_2d = channel_mapping_2d.mapping_2d(data_alpha, distribution, resolution=9)
# alpha_2d_sample = alpha_2d[0]
# show_nonzeros_2d(alpha_2d_sample)

# alpha_2d_inte = channel_mapping_2d.mapping_2d(data_alpha, distribution, resolution=9, interpolation=True, interp_method='linear')
# alpha_2d_inter_sample = alpha_2d_inte[0]
# show_nonzeros_2d(alpha_2d_inter_sample)

# 3d samples
# alpha_3d = channel_mapping_3d.mapping_3d(data_alpha, distribution, resolution=9)
# alpha_3d_sample = alpha_3d[0]
# show_nonzeros_3d(alpha_3d_sample)

# alpha_3d_inte = channel_mapping_3d.mapping_3d(data_alpha, distribution, resolution=9, interpolation=True, interp_method='linear')
# alpha_3d_inter_sample = alpha_3d_inte[0]
# show_nonzeros_3d(alpha_3d_inter_sample)

# 3d hemisphere samples
alpha_3d = channel_mapping_3d.mapping_3chs_hemisphere(data_alpha, distribution, resolution=9)
alpha_3d_sample = alpha_3d[0]
show_nonzeros_3d(alpha_3d_sample)

alpha_3d_inte = channel_mapping_3d.mapping_3chs_hemisphere(data_alpha, distribution, resolution=9, interpolation=True, interp_method='linear')
alpha_3d_inter_sample = alpha_3d_inte[0]
show_nonzeros_3d(alpha_3d_inter_sample)