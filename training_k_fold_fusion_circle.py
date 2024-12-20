# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:11:32 2024

@author: 18307
"""

import os
import numpy as np

import utils
import models
import channel_mapping_2d
import channel_mapping_3d

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

def k_fold_cross_validation(model, device, X2D, X3D, y,
                            k_folds=5, use_sequential_split=True):
    # 将数据转换为张量
    # 判断 X 和 y 是否为张量，如果是，则使用 clone().detach()，否则直接创建张量
    X2D_tensor = X2D.clone().detach().to(torch.float32) if isinstance(X2D, torch.Tensor) else torch.tensor(X2D, dtype=torch.float32)
    X3D_tensor = X3D.clone().detach().to(torch.float32) if isinstance(X3D, torch.Tensor) else torch.tensor(X3D, dtype=torch.float32)
    y_tensor = y.clone().detach().to(torch.long) if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)

    results = {}
    if use_sequential_split:
        fold_size = len(X2D_tensor) // k_folds
        indices = list(range(len(X2D_tensor)))
        
        for fold in range(k_folds):
            print(f'Fold {fold + 1}/{k_folds} (Sequential Split)')
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else len(X2D_tensor)
            val_idx = indices[val_start:val_end]
            train_idx = indices[:val_start] + indices[val_end:]
            
            X2D_train, X2D_val = X2D_tensor[train_idx], X2D_tensor[val_idx]
            X3D_train, X3D_val = X3D_tensor[train_idx], X3D_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
            
            train_loader = DataLoader(TensorDataset(X2D_train, X3D_train, y_train), batch_size=128, shuffle=True)
            val_loader = DataLoader(TensorDataset(X2D_val, X3D_val, y_val), batch_size=128, shuffle=False)
            
            results[fold] = train_and_evaluate_model(model, device, train_loader, val_loader, fold)

    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X2D_tensor)):
            print(f'Fold {fold + 1}/{k_folds} (Random Split)')
            
            X2D_train, X2D_val = X2D_tensor[train_idx], X2D_tensor[val_idx]
            X3D_train, X3D_val = X3D_tensor[train_idx], X3D_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
            
            train_loader = DataLoader(TensorDataset(X2D_train, X3D_train, y_train), batch_size=128, shuffle=True)
            val_loader = DataLoader(TensorDataset(X2D_val, X3D_val, y_val), batch_size=128, shuffle=False)
            
            results[fold] = train_and_evaluate_model(model, device, train_loader, val_loader, fold)
    
    average_accuracy = sum(results.values()) / k_folds
    print(f'\nK-Fold Cross Validation Results for {k_folds} Folds')
    print(f'Average Validation Accuracy: {average_accuracy:.2f}%\n')
    
    return average_accuracy

def train_and_evaluate_model(model, device, train_loader, val_loader, fold):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    model = model.to(device)
    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs2d, inputs3d, labels in train_loader:
            inputs2d, inputs3d, labels = inputs2d.to(device), inputs3d.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs2d, inputs3d)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs2d, inputs3d, labels in val_loader:
                inputs2d, inputs3d, labels = inputs2d.to(device), inputs3d.to(device), labels.to(device)
                
                outputs = model(inputs2d, inputs3d)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, "
        #      f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step()
    
    print(f'Fold {fold + 1} Validation Accuracy: {val_accuracy:.2f}%\n')
    return val_accuracy

# define path
path_current = os.getcwd()
path_parent = os.path.dirname(path_current)

# read channel dsitribution
path_ch_auto_distr = os.path.join(path_current, 
                                  'channel_distribution', 
                                  'biosemi62_64_channels_original_distribution.txt')
path_ch_manual_distr = os.path.join(path_current, 
                                    'channel_distribution', 
                                    'biosemi62_64_channels_manual_distribution.txt')

# initial cnn model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
model = models.CombinedModel().to(device)

# 存储结果
all_accuracies = []

# 训练遍历 sub1ex1 到 sub15ex3
for sub in range(1, 2):
    for ex in range(1, 2):
        file_name = f'sub{sub}ex{ex}.mat'
        print(f'Processing {file_name}...')
        
        dist_auto = utils.get_distribution('auto')
        dist_manual = utils.get_distribution('manual')
        
        data_alpha = utils.get_channel_feature_mat('de_LDS', 'alpha', f'sub{sub}ex{ex}')
        data_beta = utils.get_channel_feature_mat('de_LDS', 'beta', f'sub{sub}ex{ex}')
        data_gamma = utils.get_channel_feature_mat('de_LDS', 'gamma', f'sub{sub}ex{ex}')
        
        alpha_2d = channel_mapping_2d.mapping_2d(data_alpha, dist_auto, 9, interpolation=True, interp_method='linear')
        beta_2d = channel_mapping_2d.mapping_2d(data_beta, dist_auto, 9, interpolation=True, interp_method='linear')
        gamma_2d = channel_mapping_2d.mapping_2d(data_gamma, dist_auto, 9, interpolation=True, interp_method='linear')
        
        alpha_3d = channel_mapping_3d.mapping_3chs_hemisphere(data_alpha, dist_auto, 9, interpolation=True, interp_method='linear')
        beta_3d = channel_mapping_3d.mapping_3chs_hemisphere(data_beta, dist_auto, 9, interpolation=True, interp_method='linear')
        gamma_3d = channel_mapping_3d.mapping_3chs_hemisphere(data_gamma, dist_auto, 9, interpolation=True, interp_method='linear')
        
        # 对二维数据逐样本归一化
        alpha_2d = utils.safe_normalize(alpha_2d)
        beta_2d = utils.safe_normalize(beta_2d)
        gamma_2d = utils.safe_normalize(gamma_2d)
        
        # 对三维数据逐样本归一化
        alpha_3d = utils.safe_normalize(alpha_3d)
        beta_3d = utils.safe_normalize(beta_3d)
        gamma_3d = utils.safe_normalize(gamma_3d)
        
        # 堆叠数据
        data_2d = np.stack((alpha_2d, beta_2d, gamma_2d), axis=1)
        data_3d = np.stack((alpha_3d, beta_3d, gamma_3d), axis=1)

        data_2d[:] = 1
        # data_3d[:] = 1

        # 加载标签
        labels = utils.get_label()

        # K折交叉验证
        accuracies = k_fold_cross_validation(model, device, data_2d, data_3d, labels, k_folds=5, use_sequential_split=True)
        all_accuracies.append(accuracies)

# 打印所有结果
print(f'\nAll Accuracies: {all_accuracies}')
print(f'Average Accuracy: {np.mean(all_accuracies):.2f}%')