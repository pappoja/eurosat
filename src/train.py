import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from dataset import EuroSatDataset
from tqdm import tqdm
from model.resnet import ResNet50
from model.biresnet import BiResNet
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        features = batch['features'].to(device) if isinstance(model, BiResNet) else None
        
        if batch_idx == 0:
            print("Image shape:", images.shape)
            print("Features shape:", features.shape if features is not None else None)
            print("Labels:", labels.tolist())
            if features is not None:
                print("Feature sample:", features[0].cpu().numpy())
                
        optimizer.zero_grad()
        if features is not None:
            outputs = model(images, features)
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
    
    return running_loss/len(train_loader), correct/total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            features = batch['features'].to(device) if isinstance(model, BiResNet) else None
            
            if features is not None:
                outputs = model(images, features)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(val_loader), correct/total


def plot_accuracies(train_accuracies, val_accuracies, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracies')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main(data_dir, image_dir, model_type):
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
    print(f"Number of classes: {num_classes}")

    # Create model
    if model_type == 'biresnet':
        num_non_image_features = len(train_dataset.feature_columns)
        model = BiResNet(num_classes, num_non_image_features).to(device)
    else:
        model = ResNet50(num_classes).to(device)

    # Set up training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Training loop
    num_epochs = 15
    best_val_acc = 0.0
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_accuracies.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
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
            }, data_dir / 'best_model.pth')

    # Plot accuracies and save as image
    hex_id = hex(id(model))
    plot_accuracies(train_accuracies, val_accuracies, data_dir / f'accuracy_plot_{hex_id}.png')

    # Load best model and evaluate on test set
    checkpoint = torch.load(data_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Write test accuracy to results.txt
    with open(data_dir / 'results.txt', 'a') as f:
        f.write(f"{model_type}: {test_acc:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EuroSAT model.')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory where CSVs and metadata are stored')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory where image files (EuroSAT_MS) are stored')
    parser.add_argument('-m', '--model', type=str, choices=['resnet', 'biresnet'], default='resnet',
                        help='Model type to use: resnet or biresnet')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = Path(args.image_dir) if args.image_dir else data_dir

    main(data_dir, image_dir, args.model)
