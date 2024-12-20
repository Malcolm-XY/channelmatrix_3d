# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:45:40 2024

@author: 18307
"""

import math
import numpy as np
import copy
from collections import defaultdict
from scipy.interpolate import griddata

def mapping_2d(data, distribution, resolution, rounding_method='floor', interpolation=False, interp_method='linear'):
    """
    Maps data to a 2D grid matrix of size resolution^2 based on distribution coordinates.
    """
    # Deep copy inputs to ensure safety
    distribution = copy.deepcopy(distribution)
    data = np.copy(data)
    
    # Normalize or shift distribution to ensure all coordinates are within [0, 1]
    x_min, y_min = np.min(distribution['x']), np.min(distribution['y'])
    x_max, y_max = np.max(distribution['x']), np.max(distribution['y'])
    
    x_shift = -x_min if x_min < 0 else 0
    y_shift = -y_min if y_min < 0 else 0
    distribution['x'] = np.array(distribution['x']) + x_shift
    distribution['y'] = np.array(distribution['y']) + y_shift
    distribution['x'] /= (x_max + x_shift)
    distribution['y'] /= (y_max + y_shift)

    mapped_points = set()
    overlap_counts = defaultdict(int)
    
    for j in range(len(distribution['x'])):
        x = distribution['x'][j] * (resolution - 1)
        y = distribution['y'][j] * (resolution - 1)
        x, y = _apply_rounding(x, y, rounding_method)

        if 0 <= x < resolution and 0 <= y < resolution:
            if (x, y) in mapped_points:
                overlap_counts[(x, y)] += 1
            else:
                mapped_points.add((x, y))
    
    print("Mapping Results:")
    print(f"- Total grid points: {resolution * resolution}")
    print(f"- Mapped grid points: {len(mapped_points)}")
    print(f"- Overlap occurrences: {len(overlap_counts)}")
    for point, count in overlap_counts.items():
        print(f"  Overlap at grid point {point}: {count + 1} times")

    # Map the data
    data_temp = np.zeros((data.shape[0], resolution, resolution))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = distribution['x'][j] * (resolution - 1)
            y = distribution['y'][j] * (resolution - 1)
            x, y = _apply_rounding(x, y, rounding_method)
            if 0 <= x < resolution and 0 <= y < resolution:
                data_temp[i][x][y] = data[i][j]

    if interpolation:
        return fill_zeros_with_interpolation(data_temp, resolution, interp_method)
    return data_temp

def orthographic_projection_2d(data, distribution, resolution, rounding_method='floor', interpolation=False, interp_method='linear'):
    """
    Map data to a 2D grid based on distribution coordinates and apply orthographic projection.
    """
    # Deep copy inputs to ensure safety
    distribution = copy.deepcopy(distribution)
    data = np.copy(data)

    x_min, y_min = np.min(distribution['x']), np.min(distribution['y'])
    x_max, y_max = np.max(distribution['x']), np.max(distribution['y'])
    distribution['x'] = (np.array(distribution['x']) - x_min) / (x_max - x_min)
    distribution['y'] = (np.array(distribution['y']) - y_min) / (y_max - y_min)

    grid_output = np.zeros((data.shape[0], resolution, resolution))
    for i in range(data.shape[0]):
        grid = np.zeros((resolution, resolution))
        for j in range(data.shape[1]):
            x = distribution['x'][j] * (resolution - 1)
            y = distribution['y'][j] * (resolution - 1)
            x, y = _apply_rounding(x, y, rounding_method)
            if 0 <= x < resolution and 0 <= y < resolution:
                grid[x, y] += data[i, j]
        grid_output[i] = grid

    if interpolation:
        return fill_zeros_with_interpolation(grid_output, resolution, interp_method)
    return grid_output

def stereographic_projection_2d(data, distribution, resolution, rounding_method='floor', interpolation=False, interp_method='linear', prominence=0.1, epsilon=0.01):
    """
    Perform stereographic projection from 3D points to a 2D grid with optimized computation.
    """
    # Deep copy inputs to ensure safety
    distribution = copy.deepcopy(distribution)
    data = np.copy(data)

    x_coords, y_coords, z_coords = np.array(distribution['x']), np.array(distribution['y']), np.array(distribution['z'])
    z_coords = (z_coords - np.min(z_coords)) / (np.max(z_coords) - np.min(z_coords)) - prominence

    x_proj = x_coords / (1 - z_coords + epsilon)
    y_proj = y_coords / (1 - z_coords + epsilon)

    x_norm = (x_proj - np.min(x_proj)) / (np.max(x_proj) - np.min(x_proj))
    y_norm = (y_proj - np.min(y_proj)) / (np.max(y_proj) - np.min(y_proj))

    grid_output = np.zeros((data.shape[0], resolution, resolution))
    overlap_counts = defaultdict(int)
    for i in range(data.shape[0]):
        grid = np.zeros((resolution, resolution))
        mapped_points = set()
        for j in range(len(x_norm)):
            x, y = x_norm[j] * (resolution - 1), y_norm[j] * (resolution - 1)
            x, y = _apply_rounding(x, y, rounding_method)
            if 0 <= x < resolution and 0 <= y < resolution:
                if (x, y) in mapped_points:
                    overlap_counts[(x, y)] += 1
                else:
                    mapped_points.add((x, y))
                grid[x, y] += data[i, j]
        grid_output[i] = grid

    print("Overlap Statistics:")
    for point, count in overlap_counts.items():
        if count > 0:
            print(f"  Grid Point {point}: Overlapped {count + 1} times")
    
    if interpolation:
        return fill_zeros_with_interpolation(grid_output, resolution, interp_method)
    return grid_output

def _apply_rounding(x, y, method):
    if method == 'floor':
        return math.floor(x), math.floor(y)
    elif method == 'ceil':
        return math.ceil(x), math.ceil(y)
    elif method == 'round':
        return round(x), round(y)
    raise ValueError("Invalid rounding method. Choose 'floor', 'ceil', or 'round'.")

def fill_zeros_with_interpolation(data, resolution, interp_method='linear'):
    filled_data = np.copy(data)
    for sample_idx in range(data.shape[0]):
        sample = data[sample_idx]
        non_zero_coords = np.array(np.nonzero(sample)).T
        non_zero_values = sample[non_zero_coords[:, 0], non_zero_coords[:, 1]]
        grid_x, grid_y = np.mgrid[0:resolution, 0:resolution]
        grid_points = np.array([grid_x.flatten(), grid_y.flatten()]).T
        filled_values = griddata(non_zero_coords, non_zero_values, grid_points, method=interp_method, fill_value=0)
        filled_data[sample_idx] = filled_values.reshape(resolution, resolution)
    return filled_data

# %% Example Usage

# import utils
# import matplotlib.pyplot as plt

# def draw_projection(sample_projection):
#     plt.imshow(sample_projection, cmap='viridis')  # 选择颜色映射 'viridis'
#     plt.colorbar()  # 添加颜色条
#     plt.title("Matrix Visualization using imshow")
#     plt.show()

# # data and distribution
# data_alpha = utils.get_channel_feature_mat('de_LDS', 'alpha', 'sub1ex1')
# distribution = utils.get_distribution('auto')
# distribution_manual = utils.get_distribution('manual')

# # manual orthograph
# alpha_2d = mapping_2d(data_alpha, distribution_manual, 24, interpolation=False)
# #alpha_2d_interpolated = mapping_2d(data_alpha, distribution_manual, 24, interpolation=True, interp_method='linear')
# draw_projection(alpha_2d[0])
# #draw_projection(alpha_2d_interpolated[0])

# # auto orthograph
# alpha_2d_or = orthographic_projection_2d(data_alpha, distribution, 24, interpolation=False)
# #alpha_2d_or_interpolated = orthographic_projection_2d(data_alpha, distribution, 24, interpolation=True, interp_method='linear')
# draw_projection(alpha_2d_or[0])
# #draw_projection(alpha_2d_or_interpolated[0])

# # # auto stereographic
# # alpha_2d_st = stereographic_projection_2d(data_alpha, distribution, 24, interpolation=False, prominence=0.1)
# # alpha_2d_st_interpolated = stereographic_projection_2d(data_alpha, distribution, 24, interpolation=True, interp_method='linear', prominence=0.1)
# # draw_projection(alpha_2d_st[0])
# # draw_projection(alpha_2d_st_interpolated[0])