import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, features)
        self.enc2 = self.conv_block(features, features*2)
        self.enc3 = self.conv_block(features*2, features*4)
        self.enc4 = self.conv_block(features*4, features*8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(features*8, features*16)
        
        # Decoder
        # After upsampling: up4 outputs features*8, concatenate with e4 (features*8) = features*8 + features*8
        self.dec4 = self.conv_block(features*8 + features*8, features*8)
        # After upsampling: up3 outputs features*4, concatenate with e3 (features*4) = features*4 + features*4  
        self.dec3 = self.conv_block(features*4 + features*4, features*4)
        # After upsampling: up2 outputs features*2, concatenate with e2 (features*2) = features*2 + features*2
        self.dec2 = self.conv_block(features*2 + features*2, features*2)
        # After upsampling: up1 outputs features, concatenate with e1 (features) = features + features
        self.dec1 = self.conv_block(features + features, features)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Upsampling
        self.up4 = nn.ConvTranspose2d(features*16, features*8, 2, 2)
        self.up3 = nn.ConvTranspose2d(features*8, features*4, 2, 2)
        self.up2 = nn.ConvTranspose2d(features*4, features*2, 2, 2)
        self.up1 = nn.ConvTranspose2d(features*2, features, 2, 2)
        
        # Final conv for segmentation-like output
        self.final_conv = nn.Conv2d(features, out_channels, 1)
        
        # For classification
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, 1)
    
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        out = self.final_conv(d1)
        
        # For classification
        pooled = self.global_avg_pool(out).squeeze(-1).squeeze(-1)
        logits = self.classifier(pooled)
        return logits