import torch
import torch.nn as nn
import timm

class PretrainedTreeClassifier(nn.Module):
    """
    Pretrained model for tree classification using timm.
    Supports Swin-Tiny, ConvNeXt-Tiny, and EfficientNet models.
    """
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=2, img_size=128, dropout=0.3):
        super(PretrainedTreeClassifier, self).__init__()
        
        # Create pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove default classifier
            in_chans=3,
            img_size=img_size
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            # Handle different output shapes
            if len(features.shape) > 2:
                # Spatial features - will be pooled in forward
                feature_dim = features.shape[1]
            else:
                # Already pooled features
                feature_dim = features.shape[-1]
        
        # Custom classifier head for binary classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1)  # Binary classification
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Global average pooling if needed (some timm models return spatial features)
        if len(features.shape) > 2:
            features = torch.mean(features, dim=(2, 3))
        # Ensure features are 1D
        if len(features.shape) > 1:
            features = features.view(features.size(0), -1)
        # Classify
        logits = self.classifier(features)
        return logits.squeeze(-1)

