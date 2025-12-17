import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import argparse

from unet import UNet
from pretrained_model import PretrainedTreeClassifier

import albumentations as A
from albumentations.pytorch import ToTensorV2

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.67, gamma=2.0, reduction='mean'):
        """
        Focal loss
        alpha: Weighting factor for the rare class (usually positive).
        gamma: Focusing parameter. Higher values (e.g., 2.0) reduce loss for easy examples.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (raw output from model)
        # targets: binary labels (0 or 1)
        
        # Calculate BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get the probabilities (pt)
        pt = torch.exp(-bce_loss)
        
        # Calculate Focal Loss
        # Formula: -alpha * (1-pt)^gamma * log(pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TreeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.tensor(label, dtype=torch.float32)

# def get_transforms(is_train=True, img_size=128, 
#                    norm_mean=[0.273, 0.337, 0.274], 
#                    norm_std=[0.131, 0.124, 0.108]):
    
#     base_transforms = [
#         A.Resize(img_size, img_size),
#         A.Normalize(mean=norm_mean, std=norm_std),
#         ToTensorV2()
#     ]
    
#     if not is_train:
#         return A.Compose(base_transforms)
    
#     # Insert augmentations BEFORE normalization
#     augmentations = [
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.ShiftScaleRotate(
#             shift_limit=0.0625,
#             scale_limit=0.1,
#             rotate_limit=15,
#             p=0.5
#         ),
#         A.RandomBrightnessContrast(
#             brightness_limit=0.2,
#             contrast_limit=0.2,
#             p=0.5
#         ),
#     ]
    
#     return A.Compose(augmentations + base_transforms)

def get_transforms(is_train=True, img_size=128, 
                   norm_mean=[0.273, 0.337, 0.274], 
                   norm_std=[0.131, 0.124, 0.108]):
    
    # 1. Resize/Normalize happens at the end
    base_transforms = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2()
    ]
    
    if not is_train:
        return A.Compose(base_transforms)
    
    # 2. Augmentations
    augmentations = [
        # --- GEOMETRIC ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            # CRITICAL: Use reflection to fill empty corners instead of black
            border_mode=cv2.BORDER_REFLECT_101, 
            p=0.5
        ),

        # --- SENSOR SIMULATION (New) ---
        # Helps preventing overfitting to specific camera quality
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.MultiplicativeNoise(multiplier=[0.8, 1.2], p=0.5),
        ], p=0.3),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.MotionBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.2),

        # --- COLOR ---
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
    ]
    
    # Note: Augmentations happen BEFORE Resize. 
    # This is good for quality, but slower if input images are huge.
    return A.Compose(augmentations + base_transforms)

def train_model(model_type='pretrained', model_name='swin_tiny_patch4_window7_224', img_size=128, 
                batch_size=16, num_epochs=50, lr=1e-4):
    """
    Train tree classification model.
    
    Args:
        model_type: 'pretrained' or 'unet'
        model_name: Model name for pretrained (e.g., 'swin_tiny_patch4_window7_224', 
                    'convnext_tiny.fb_in22k', 'tf_efficientnet_b1.ns_jft_in1k')
        img_size: Image size for training (128 for pretrained, 256 for UNet)
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}')
    print(f'Model type: {model_type}')
    if model_type == 'pretrained':
        print(f'Pretrained model: {model_name}')
    
    # Load data
    data_dir = Path('src/data/raw')
    labels_df = pd.read_csv(data_dir / 'labels_llm_toloka.csv')
    image_dir = data_dir / 'images_upscaled'
    
    image_paths = [image_dir / fname for fname in labels_df['filename']]
    labels = labels_df['label'].values
    
    # Split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.1, random_state=42, stratify=temp_labels
    )
    
    print(f'Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}')
    
    # Datasets
    train_dataset = TreeDataset(train_paths, train_labels, get_transforms(is_train=True, img_size=img_size))
    val_dataset = TreeDataset(val_paths, val_labels, get_transforms(is_train=False, img_size=img_size))
    test_dataset = TreeDataset(test_paths, test_labels, get_transforms(is_train=False, img_size=img_size))
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    if model_type == 'pretrained':
        model = PretrainedTreeClassifier(model_name=model_name, num_classes=2, img_size=img_size).to(device)
        # Use different learning rates for backbone and classifier (fine-tuning)
        backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
        classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for pretrained weights
            {'params': classifier_params, 'lr': lr}       # Higher LR for new classifier
        ], weight_decay=0.01)
    else:
        model = UNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss()
    # train_labels = np.array(train_labels)  # ensure it's array-like
    # num_pos = train_labels.sum()
    # num_neg = len(train_labels) - num_pos

    # if num_pos == 0:
    #     raise ValueError("No positive samples in training set!")
    # if num_neg == 0:
    #     raise ValueError("No negative samples in training set!")

    # pos_weight = num_neg / num_pos
    # print(f"Class balance: {num_neg} negatives, {num_pos} positives → pos_weight = {pos_weight:.3f}")

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    
    # Load checkpoint if exists to continue training
    checkpoint_path = Path(f'src/model/best_model_{model_type}_{model_name if model_type == "pretrained" else "unet"}.pth')
    if checkpoint_path.exists():
        print(f'Found checkpoint at {checkpoint_path}')
        print('Loading checkpoint to continue training from fine-tuned weights...')
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print('✓ Checkpoint loaded successfully - continuing training from fine-tuned model')
        except Exception as e:
            print(f'⚠ Warning: Could not load checkpoint ({e}). Starting from pretrained weights instead.')
    else:
        print('No checkpoint found - starting from ImageNet pretrained weights')
    
    # Tensorboard
    run_name = f'tree_classifier_{model_type}_{model_name if model_type == "pretrained" else "unet"}'
    writer = SummaryWriter(f'runs/{run_name}')
    
    # Training
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs.squeeze()) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            model_path = f'src/model/best_model_{model_type}_{model_name if model_type == "pretrained" else "unet"}.pth'
            torch.save(model.state_dict(), model_path)
            print(f'  -> Saved best model (Val Acc: {val_acc:.4f})')
    
    # Test
    model_path = f'src/model/best_model_{model_type}_{model_name if model_type == "pretrained" else "unet"}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs.squeeze()) > 0.5
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = test_correct / test_total
    print('\n' + '='*50)
    print('Final Results:')
    print(f'Best Val Loss: {best_val_loss:.4f}')
    print(f'Best Val Acc: {best_val_acc:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'{"="*50}')
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train tree classification model')
    parser.add_argument('--model', type=str, default='pretrained', 
                       choices=['pretrained', 'unet'],
                       help='Model type: pretrained or unet')
    parser.add_argument('--model_name', type=str, default='swin_tiny_patch4_window7_224',
                       help='Pretrained model name (for pretrained type). Options: swin_tiny_patch4_window7_224, convnext_tiny.fb_in22k, tf_efficientnet_b1.ns_jft_in1k')
    parser.add_argument('--img_size', type=int, default=128,
                       help='Image size (128 for pretrained, 256 for UNet)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Adjust img_size based on model type if not specified
    if args.model == 'unet' and args.img_size == 128:
        args.img_size = 256
    
    train_model(
        model_type=args.model,
        model_name=args.model_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr
    )