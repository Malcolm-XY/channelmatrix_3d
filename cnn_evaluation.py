# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:14:42 2024

@author: usouu
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

def train_model(model, train_loader, device, optimizer, criterion, scheduler, epochs=30):
    """
    Train a PyTorch model.

    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        device (torch.device): Device to use for training.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (torch.nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epochs (int, optional): Number of training epochs. Default is 30.

    Returns:
        torch.nn.Module: The trained model.
    """
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch_2D, batch_3D, batch_labels in train_loader:
            batch_2D, batch_3D, batch_labels = (
                batch_2D.to(device),
                batch_3D.to(device),
                batch_labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(batch_2D, batch_3D)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    return model

def test_model(model, val_loader, device, criterion):
    """
    Test a PyTorch model and calculate accuracy.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to use for testing.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Validation accuracy.
    """
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_2D, batch_3D, batch_labels in val_loader:
            batch_2D, batch_3D, batch_labels = (
                batch_2D.to(device),
                batch_3D.to(device),
                batch_labels.to(device),
            )

            outputs = model(batch_2D, batch_3D)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    print(f"Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.4%}")

    return accuracy

def k_fold_evaluation(model, X2D, X3D, y, k_folds=5, batch_size=128, 
                      use_sequential_split=True, epochs=30, learning_rate=0.0005, 
                      weight_decay=1e-4, step_size=5, gamma=0.5):
    """
    Perform k-fold cross-validation on a PyTorch model.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        X2D (array-like or torch.Tensor): 2D input data.
        X3D (array-like or torch.Tensor): 3D input data.
        y (array-like or torch.Tensor): Labels.
        k_folds (int, optional): Number of folds for cross-validation. Default is 5.
        batch_size (int, optional): Batch size for DataLoader. Default is 128.
        use_sequential_split (bool, optional): Use sequential split if True, otherwise random split. Default is True.
        epochs (int, optional): Number of training epochs. Default is 30.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.0005.
        weight_decay (float, optional): Weight decay (L2 regularization). Default is 1e-4.
        step_size (int, optional): Step size for the learning rate scheduler. Default is 5.
        gamma (float, optional): Multiplicative factor of learning rate decay. Default is 0.5.

    Returns:
        float: Average validation accuracy across all folds.
    """
    # Convert data to tensors
    X2D_tensor = torch.as_tensor(X2D, dtype=torch.float32)
    X3D_tensor = torch.as_tensor(X3D, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    if use_sequential_split:
        fold_size = len(X2D_tensor) // k_folds
        indices = list(range(len(X2D_tensor)))

        for fold in range(k_folds):
            print(f"Fold {fold + 1}/{k_folds} (Sequential Split)")
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else len(X2D_tensor)
            val_idx = indices[val_start:val_end]
            train_idx = indices[:val_start] + indices[val_end:]

            X2D_train, X2D_val = X2D_tensor[train_idx], X2D_tensor[val_idx]
            X3D_train, X3D_val = X3D_tensor[train_idx], X3D_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

            train_loader = DataLoader(TensorDataset(X2D_train, X3D_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X2D_val, X3D_val, y_val), batch_size=batch_size, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            trained_model = train_model(model, train_loader, device, optimizer, criterion, scheduler, epochs)
            fold_accuracy = test_model(trained_model, val_loader, device, criterion)
            results.append(fold_accuracy)

    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X2D_tensor)):
            print(f"Fold {fold + 1}/{k_folds} (Random Split)")

            X2D_train, X2D_val = X2D_tensor[train_idx], X2D_tensor[val_idx]
            X3D_train, X3D_val = X3D_tensor[train_idx], X3D_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

            train_loader = DataLoader(TensorDataset(X2D_train, X3D_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X2D_val, X3D_val, y_val), batch_size=batch_size, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            trained_model = train_model(model, train_loader, device, optimizer, criterion, scheduler, epochs)
            fold_accuracy = test_model(trained_model, val_loader, device, criterion)
            results.append(fold_accuracy)

    average_accuracy = sum(results) / k_folds
    print(f"\nK-Fold Cross Validation Results for {k_folds} Folds")
    print(f"Average Validation Accuracy: {average_accuracy:.2f}%\n")

    return average_accuracy
