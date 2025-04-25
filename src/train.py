import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from dataset import EuroSatDataset
from tqdm import tqdm
from model.resnet import ResNet
from model.biresnet import BiResNet
from model.simplecnn import SimpleCNN
from model.filmresnet import FiLMResNet
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_idx = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        features = batch['features'].to(device) if not isinstance(model, ResNet) else None
        country_idx = batch['country_idx'].to(device) if 'country_idx' in batch and batch['country_idx'] is not None else None
        
        optimizer.zero_grad()
        if isinstance(model, BiResNet) or isinstance(model, FiLMResNet):
            outputs = model(images, country_idx, features)
        elif isinstance(model, SimpleCNN):
            outputs = model(images, features, country_idx)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        batch_idx += 1
    
    return running_loss/len(train_loader), correct/total


def validate(model, val_loader, criterion, device, label_to_idx, model_type, input):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            features = batch['features'].to(device) if isinstance(model, BiResNet) or isinstance(model, SimpleCNN) else None
            country_idx = batch['country_idx'].to(device) if 'country_idx' in batch and batch['country_idx'] is not None else None

            if isinstance(model, BiResNet) or isinstance(model, FiLMResNet):
                outputs = model(images, country_idx, features)
            elif isinstance(model, SimpleCNN):
                outputs = model(images, features, country_idx)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Plot confusion matrix
    classes = list(label_to_idx.keys())
    plot_confusion_matrix(y_true, y_pred, classes, "../results", model_type, input, normalize=True, title='Validation Confusion Matrix')

    return running_loss/len(val_loader), correct/total


def plot_accuracies(train_accuracies, val_accuracies, save_path, model_type, input):
    plt.figure(figsize=(10, 5))
    
    # Shift x-axis to start from 1
    epochs = np.arange(1, len(train_accuracies) + 1)

    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')

    best_epoch = int(np.argmax(val_accuracies)) + 1
    best_val_acc = val_accuracies[best_epoch - 1]

    plt.axvline(best_epoch, color='black', linestyle='--', 
                label=f'Best validation accuracy: {best_val_acc:.2%}')
    
    # Set input type for title
    if input == 'image':
        input_type = "image-only"
    elif input == 'image_country':
        input_type = "image+country"
    else:
        input_type = "all data"

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_type} ({input_type}): Train and Validation Accuracies')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs[::5])
    plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    save_path = Path(save_path)
    plt.savefig(save_path / f'{model_type}_{input}_training.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path, model_type, input, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    save_path = Path(save_path)
    plt.savefig(save_path / f'{model_type}_{input}_confusion.png')
    plt.close()


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def main(data_dir, image_dir, model_type, input, num_epochs):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    csv_data_dir = data_dir / 'csv_data'
    
    # Create a unified label_to_idx mapping
    train_df = pd.read_csv(csv_data_dir/'train_index.csv')
    val_df = pd.read_csv(csv_data_dir/'val_index.csv')
    test_df = pd.read_csv(csv_data_dir/'test_index.csv')
    all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']]).unique()
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    train_dataset = EuroSatDataset(csv_data_dir/'train_index.csv', root_dir=image_dir, label_to_idx=label_to_idx)
    val_dataset = EuroSatDataset(csv_data_dir/'val_index.csv', root_dir=image_dir, label_to_idx=label_to_idx)
    test_dataset = EuroSatDataset(csv_data_dir/'test_index.csv', root_dir=image_dir, label_to_idx=label_to_idx)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Get number of classes
    num_classes = len(train_dataset.label_to_idx)

    # Extract the number of countries for defining the embedding layer
    max_country_idx = max(train_dataset.data_frame['country_id'].max(),
                          val_dataset.data_frame['country_id'].max(),
                          test_dataset.data_frame['country_id'].max())
    num_countries = int(max_country_idx + 1)

    # Create model
    if model_type in ['biresnet18', 'biresnet50']:
        num_non_image_features = len(train_dataset.feature_columns)
        model = BiResNet(model_type, num_classes, num_non_image_features, num_countries, input_type=input).to(device)
    elif model_type in ['resnet18', 'resnet50']:
        model = ResNet(model_type, num_classes).to(device)
    elif model_type in ['filmresnet18', 'filmresnet50']:
        num_non_image_features = len(train_dataset.feature_columns)
        model = FiLMResNet(model_type, num_classes, num_non_image_features, num_countries, input_type=input).to(device)
    elif model_type == 'simplecnn':
        num_non_image_features = len(train_dataset.feature_columns)
        model = SimpleCNN(num_classes, num_non_image_features, num_countries, input_type=input).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: 'biresnet18', 'biresnet50', 'resnet50', 'resnet18', 'simplecnn'")

    # Set up training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Training loop
    best_val_acc = 0.0
    train_accuracies = []
    val_accuracies = []
    early_stopping = EarlyStopping(patience=10)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_accuracies.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, label_to_idx, model_type, input)
        val_accuracies.append(val_acc)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, data_dir / f'best_{model_type}_{input}.pth')

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Plot accuracies and save as image
    plot_accuracies(train_accuracies, val_accuracies, "../results", model_type, input)

    # Evaluate on test set
    test_dataset = EuroSatDataset(csv_data_dir/'test_index.csv', root_dir=image_dir, label_to_idx=label_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            features = batch['features'].to(device) if not isinstance(model, ResNet) else None
            country_idx = batch['country_idx'].to(device) if 'country_idx' in batch and batch['country_idx'] is not None else None

            if isinstance(model, BiResNet) or isinstance(model, FiLMResNet):
                outputs = model(images, country_idx, features)
            elif isinstance(model, SimpleCNN):
                outputs = model(images, features, country_idx)
            else:
                outputs = model(images)

            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Plot confusion matrix
    classes = list(label_to_idx.keys())
    plot_confusion_matrix(y_true, y_pred, classes, "../results", model_type, input, normalize=True)

    # Load best model and evaluate on test set
    checkpoint = torch.load(data_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device, label_to_idx, model_type, input)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Write test accuracy to results.txt
    with open(data_dir / 'results.txt', 'a') as f:
        f.write(f"{model_type} ({input}): {test_acc:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EuroSAT model.')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory where CSVs and metadata are stored')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory where image files (EuroSAT_MS) are stored')
    parser.add_argument('-m', '--model', type=str, choices=['resnet18', 'resnet50', 'biresnet18', 'biresnet50', 'simplecnn'], default='resnet18',
                        help='Model type to use: resnet18, resnet50, biresnet18, biresnet50, or simplecnn')
    parser.add_argument('--input', type=str, choices=['image', 'image_country', 'image_country_all'], default='image',
                        help='Input type to use: image, image_country, or image_country_all')
    parser.add_argument('--n-epochs', type=int, default=20, help='Number of epochs to train for')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = Path(args.image_dir) if args.image_dir else data_dir

    main(data_dir, image_dir, args.model, args.input, args.n_epochs)
