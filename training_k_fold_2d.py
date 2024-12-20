# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:11:32 2024

@author: 18307
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import KFold

import utils
import models
import models_multiscale
import channel_mapping_2d

def k_fold_cross_validation(model, X, Y, k_folds=5, use_sequential_split=True):
    X_tensor = X.clone().detach().to(torch.float32) if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    Y_tensor = Y.clone().detach().to(torch.int64) if isinstance(Y, torch.Tensor) else torch.tensor(Y, dtype=torch.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    results = []
    if use_sequential_split:
        fold_size = len(X_tensor) // k_folds
        indices = list(range(len(X_tensor)))

        for fold in range(k_folds):
            print(f'Fold {fold + 1}/{k_folds} (Sequential Split)')
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else len(X_tensor)
            val_idx = indices[val_start:val_end]
            train_idx = indices[:val_start] + indices[val_end:]

            X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
            Y_train, Y_val = Y_tensor[train_idx], Y_tensor[val_idx]

            train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=128, shuffle=False)

            result = train_and_evaluate_model(model, device, train_loader, val_loader, fold)
            results.append(result)

    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tensor)):
            print(f'Fold {fold + 1}/{k_folds} (Random Split)')

            X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
            Y_train, Y_val = Y_tensor[train_idx], Y_tensor[val_idx]

            train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=128, shuffle=False)

            result = train_and_evaluate_model(model, device, train_loader, val_loader, fold)
            results.append(result)

    avg_results = {
        'accuracy': np.mean([res['accuracy'] for res in results]),        
        'loss': np.mean([res['loss'] for res in results]),
        'recall': np.mean([res['recall'] for res in results]),
        'f1_score': np.mean([res['f1_score'] for res in results]),
    }

    print(f"{k_folds}-Fold Cross Validation Results:")
    print(f"Average Accuracy: {avg_results['accuracy']:.2f}%")    
    print(f"Average Loss: {avg_results['loss']:.4f}")
    print(f"Average Recall: {avg_results['recall']:.2f}%")
    print(f"Average F1 Score: {avg_results['f1_score']:.2f}%\n")

    return avg_results

def train_and_evaluate_model(model, device, train_loader, val_loader, fold):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model = model.to(device)
    epochs = 30

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100

    print(f'Fold {fold + 1} Validation Accuracy: {accuracy:.2f}%, Loss: {val_loss / len(val_loader):.4f}, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%\n')
    return {
        'accuracy': accuracy,        
        'loss': val_loss / len(val_loader),
        'recall': recall,
        'f1_score': f1
    }

def k_fold_evaluation_circle(model, subject_range, experiment_range, feature):
    labels = utils.get_label()
    distribution = utils.get_distribution('auto')
    
    results_entry = []
    for sub in subject_range:
        for ex in experiment_range:
            file_name = f'sub{sub}ex{ex}.mat'
            print(f'Processing {file_name}...')
            data_alpha = utils.get_channel_feature_mat('de_LDS', 'alpha', f'sub{sub}ex{ex}')
            data_beta = utils.get_channel_feature_mat('de_LDS', 'beta', f'sub{sub}ex{ex}')
            data_gamma = utils.get_channel_feature_mat('de_LDS', 'gamma', f'sub{sub}ex{ex}')

            # alpha_2d = channel_mapping_2d.mapping_2d(data_alpha, distribution, 9, interpolation=True, interp_method='linear')
            # beta_2d = channel_mapping_2d.mapping_2d(data_beta, distribution, 9, interpolation=True, interp_method='linear')
            # gamma_2d = channel_mapping_2d.mapping_2d(data_gamma, distribution, 9, interpolation=True, interp_method='linear')

            alpha_2d = channel_mapping_2d.stereographic_projection_2d(data_alpha, distribution, 24, interpolation=True, interp_method='linear')
            beta_2d = channel_mapping_2d.stereographic_projection_2d(data_beta, distribution, 24, interpolation=True, interp_method='linear')
            gamma_2d = channel_mapping_2d.stereographic_projection_2d(data_gamma, distribution, 24, interpolation=True, interp_method='linear')

            alpha_2d = utils.safe_normalize(alpha_2d)
            beta_2d = utils.safe_normalize(beta_2d)
            gamma_2d = utils.safe_normalize(gamma_2d)
            
            data_2d = np.stack((alpha_2d, beta_2d, gamma_2d), axis=1)
            
            result = k_fold_cross_validation(model, data_2d, labels, use_sequential_split=True)
            
            # Add identifier to the result
            result['Identifier'] = f'sub{sub}ex{ex}'
            results_entry.append(result)

    # print(f'Final Results: {results_entry}')
    print('K-Fold Validation compelete\n')
    
    return results_entry

# usage
# model = models.EnhancedCNN2DModel2(channels=3)
model = models_multiscale.MultiScaleCNN_1()
feature = 'de_LDS'
subject_range, experiment_range = range(1, 4), range(1, 4)
results = k_fold_evaluation_circle(model, subject_range, experiment_range, feature)

# Save to xlsx
results_df = pd.DataFrame(results)
columns_order = ['Identifier'] + [col for col in results_df.columns if col != 'Identifier']
results_df = results_df[columns_order]
output_path = os.path.join(os.getcwd(),'k_fold_results.xlsx')
results_df.to_excel(output_path, index=False, sheet_name='K-Fold Results')