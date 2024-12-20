# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:45:40 2024

@author: 18307
"""

import math
import numpy as np
from collections import defaultdict
from scipy.interpolate import griddata

def mapping_3d(data, distribution, resolution, rounding_method='floor', interpolation=False, interp_method='linear'):
    """
    Maps data to a 3D grid matrix of size resolution^3 based on distribution coordinates.

    Args:
    - data (ndarray): Input data of shape (N, M), where N is the number of samples and M is the number of features.
    - distribution (dict): Dictionary with 'x', 'y', and 'z' coordinates for each feature.
    - resolution (int): The size of the grid (resolution x resolution x resolution).
    - interpolation (bool): Whether to interpolate the grid to fill zeros.
    - interp_method (str): Interpolation method to use ('linear' by default).
    - rounding_method (str): Rounding method to use ('floor', 'ceil', or 'round').

    Returns:
    - ndarray: 3D mapped data, optionally interpolated.
    """
    # Normalize or shift distribution to ensure all coordinates are within [0, 1]
    x_min, y_min, z_min = np.min(distribution['x']), np.min(distribution['y']), np.min(distribution['z'])
    x_max, y_max, z_max = np.max(distribution['x']), np.max(distribution['y']), np.max(distribution['z'])

    # Shift coordinates to be >= 0
    x_shift = -x_min if x_min < 0 else 0
    y_shift = -y_min if y_min < 0 else 0
    z_shift = -z_min if z_min < 0 else 0
    distribution['x'] = np.array(distribution['x']) + x_shift
    distribution['y'] = np.array(distribution['y']) + y_shift
    distribution['z'] = np.array(distribution['z']) + z_shift

    # Normalize coordinates to range [0, 1]
    distribution['x'] = distribution['x'] / (x_max + x_shift)
    distribution['y'] = distribution['y'] / (y_max + y_shift)
    distribution['z'] = distribution['z'] / (z_max + z_shift)

    # Prepare to detect squeeze and overlap
    mapped_points = set()  # Set of unique mapped grid points
    overlap_counts = defaultdict(int)  # Count of overlaps at each grid point

    for j in range(len(distribution['x'])):
        x = distribution['x'][j] * (resolution - 1)
        y = distribution['y'][j] * (resolution - 1)
        z = distribution['z'][j] * (resolution - 1)

        # Apply rounding method
        if rounding_method == 'floor':
            x, y, z = math.floor(x), math.floor(y), math.floor(z)
        elif rounding_method == 'ceil':
            x, y, z = math.ceil(x), math.ceil(y), math.ceil(z)
        elif rounding_method == 'round':
            x, y, z = round(x), round(y), round(z)
        else:
            raise ValueError("Invalid rounding method. Choose 'floor', 'ceil', or 'round'.")

        # Ensure indices are within bounds
        if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
            if (x, y, z) in mapped_points:
                overlap_counts[(x, y, z)] += 1
            else:
                mapped_points.add((x, y, z))
        else:
            print(f"Invalid index: x={x}, y={y}, z={z} for feature {j}")

    # Calculate squeeze
    total_grid_points = resolution ** 3
    mapped_count = len(mapped_points)
    squeeze_count = total_grid_points - mapped_count

    # Report mapping results
    print("Mapping Results:")
    print(f"- Total grid points: {total_grid_points}")
    print(f"- Mapped grid points: {mapped_count}")
    print(f"- Squeeze (unmapped points): {squeeze_count}")
    print(f"- Overlap occurrences: {len(overlap_counts)}")
    for point, count in overlap_counts.items():
        print(f"  Overlap at grid point {point}: {count + 1} times")

    # Now map the data for each sample
    data_temp = np.zeros((data.shape[0], resolution, resolution, resolution))
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = distribution['x'][j] * (resolution - 1)
            y = distribution['y'][j] * (resolution - 1)
            z = distribution['z'][j] * (resolution - 1)
            
            # Apply rounding method
            if rounding_method == 'floor':
                x, y, z = math.floor(x), math.floor(y), math.floor(z)
            elif rounding_method == 'ceil':
                x, y, z = math.ceil(x), math.ceil(y), math.ceil(z)
            elif rounding_method == 'round':
                x, y, z = round(x), round(y), round(z)

            # Ensure indices are within bounds
            if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
                data_temp[i][int(x)][int(y)][int(z)] = data[i][j]

    if interpolation:
        # Fill zeros with specified interpolation method (to be implemented)
        data_filled = fill_zeros_with_interpolation_3d(data_temp, resolution, interp_method)
        print('Dataset 3D Mapping Done')
        return data_filled
    else:
        print('Dataset 3D Mapping Done')
        return data_temp

def fill_zeros_with_interpolation_3d(data, resolution, interp_method):
    filled_data = np.copy(data)  # Copy the data for filling
    
    for sample_idx in range(data.shape[0]):
        sample = data[sample_idx]
        
        # Get non-zero value coordinates
        non_zero_coords = np.array(np.nonzero(sample)).T
        non_zero_values = sample[non_zero_coords[:, 0], non_zero_coords[:, 1], non_zero_coords[:, 2]]
        
        # Get all the coordinates in the 3D grid
        grid_x, grid_y, grid_z = np.mgrid[0:resolution, 0:resolution, 0:resolution]
        grid_points = np.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T
        
        # Perform 3D interpolation for zero values
        filled_values = griddata(non_zero_coords, non_zero_values, grid_points, method=interp_method, fill_value=0)
        
        # Reshape back to the original 3D shape and fill the sample
        filled_sample = filled_values.reshape(resolution, resolution, resolution)
        filled_data[sample_idx] = filled_sample
    
    return filled_data

def mapping_3chs_hemisphere(data, distribution, resolution, rounding_method='floor', interpolation=False, interp_method='linear'):
    # Normalize or shift distribution to ensure all coordinates are within [0, 1]
    x_min, y_min, z_min = np.min(distribution['x']), np.min(distribution['y']), np.min(distribution['z'])
    x_max, y_max, z_max = np.max(distribution['x']), np.max(distribution['y']), np.max(distribution['z'])

    # Shift coordinates to be >= 0
    x_shift = -x_min if x_min < 0 else 0
    y_shift = -y_min if y_min < 0 else 0
    z_shift = -z_min if z_min < 0 else 0
    distribution['x'] = np.array(distribution['x']) + x_shift
    distribution['y'] = np.array(distribution['y']) + y_shift
    distribution['z'] = np.array(distribution['z']) + z_shift
    
    # Normalize coordinates to range [0, 1]
    distribution['x'] = distribution['x'] / (x_max + x_shift)
    distribution['y'] = distribution['y'] / (y_max + y_shift)
    distribution['z'] = distribution['z'] / (z_max + z_shift)
    
    # Adjust Z axis to map to [resolution/2 + 1, resolution]
    z_offset = resolution // 2 + 1
    distribution['z'] = distribution['z'] * (resolution / 2 - 1) + z_offset
    
    # Now map the data for each sample
    data_temp = np.zeros((data.shape[0], resolution, resolution, resolution))
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = distribution['x'][j] * (resolution - 1)
            y = distribution['y'][j] * (resolution - 1)
            z = distribution['z'][j]  # Already adjusted to [resolution/2 + 1, resolution]
            
            # Apply rounding method
            if rounding_method == 'floor':
                x, y, z = math.floor(x), math.floor(y), math.floor(z)
            elif rounding_method == 'ceil':
                x, y, z = math.ceil(x), math.ceil(y), math.ceil(z)
            elif rounding_method == 'round':
                x, y, z = round(x), round(y), round(z)
                        
            # Ensure indices are within bounds
            if 0 <= x < resolution and 0 <= y < resolution and z_offset <= z <= resolution:
                data_temp[i][int(x)][int(y)][int(z)] = data[i][j]

    # Optionally apply interpolation if needed
    if interpolation:
        data_filled = fill_zeros_with_interpolation_3d(data_temp, resolution, interp_method)
        print('Dataset 3D Mapping Done')
        return data_filled
    else:
        return data_temp    

def fill_zeros_with_interpolation_on_hemisphere(data, center=(4, 4, 4), radius1=4.5, radius2=3.5):
    """
    对 EEG 数据进行三维插值，仅填充头皮半球范围内的零值点。
    
    :param data: 原始 3D 数据，形状为 (N, 9, 9, 9)。
    :param center: 半球中心点坐标，默认为 (4, 4, 4)。
    :param radius: 半球半径，默认为 4.5。
    :return: 填充后的 3D 数据。
    """
    filled_data = np.copy(data)  # 复制一份数据，用于插值

    # 定义一个函数来检查点是否在头皮半球内
    def is_in_hemisphere(x, y, z, center, radius1, radius2):
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        # 检查点是否在球内且位于上半部分
        return distance <= radius1 and distance >= radius2 and z >= center[2]

    for sample_idx in range(data.shape[0]):
        sample = data[sample_idx]
        
        # 获取非零值的坐标
        non_zero_coords = np.array(np.nonzero(sample)).T
        non_zero_values = sample[non_zero_coords[:, 0], non_zero_coords[:, 1], non_zero_coords[:, 2]]
        
        # 获取所有的坐标
        grid_x, grid_y, grid_z = np.mgrid[0:9, 0:9, 0:9]
        grid_points = np.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

        # 对零值部分进行三维插值
        filled_values = griddata(non_zero_coords, non_zero_values, grid_points, method='nearest', fill_value=0)
        
        # 重新 reshape 成原始的三维形状
        filled_sample = filled_values.reshape(9, 9, 9)

        # 遍历每个点，检查是否在半球范围内，若不在则置为零
        for x in range(9):
            for y in range(9):
                for z in range(9):
                    if not is_in_hemisphere(x, y, z, center, radius1, radius2):
                        filled_sample[x, y, z] = 0

        filled_data[sample_idx] = filled_sample

    return filled_data