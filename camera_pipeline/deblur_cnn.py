import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MobileNetDeblur(nn.Module):
    """
    Lightweight MobileNet-based Encoder-Decoder for deblurring.
    """
    def __init__(self, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        
        # Encoder (Simplified MobileNetV2-like)
        self.enc1 = ConvBlock(3, 16, stride=1)
        self.enc2 = ConvBlock(16, 32, stride=2) 
        self.enc3 = ConvBlock(32, 64, stride=2)
        
        # Bottleneck
        self.neck = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64)
        )
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(64 + 32, 32) # Skip connection
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(32 + 16, 16) # Skip connection
        
        self.final = nn.Conv2d(16, 3, 3, 1, 1)
        
        # Dummy sharpening kernel for prototype demo if not trained
        self._init_dummy_sharpen()

    def _init_dummy_sharpen(self):
        # A 3x3 sharpening kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        
        # Reshape for 3 channels (depthwise)
        # Shape: (out_channels, in_channels/groups, k, k)
        # We want to apply this to R, G, B independently.
        self.dummy_weight = torch.from_numpy(kernel).view(1, 1, 3, 3).repeat(3, 1, 1, 1).cuda() if torch.cuda.is_available() else torch.from_numpy(kernel).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def forward(self, x):
        """
        x: (B, 3, H, W) normalized [0, 1]
        """
        # If we are training, OR if we have loaded weights, use the Neural Network
        if self.training or self.pretrained:
            return self._forward_net(x)
        else:
            # Prototype Mode (Heuristic)
            return self._forward_heuristic(x)

    def _forward_heuristic(self, x):
        """
        Prototype heuristic logic.
        """
        # 1. Convert to Numpy (H, W, 3)
        # Handle batch size > 1 issue? Prototype is usually BS=1
        if x.size(0) > 1:
            # Just return input for batch > 1 in heuristic mode to avoid errors
            return x
            
        img_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 2. Large-Scale Unsharp Mask (for defocus recovery)
        # Create a very blurred version to extract large-scale contrast
        large_blur = cv2.GaussianBlur(img_np, (0, 0), 10.0)
        large_details = cv2.addWeighted(img_np, 1.5, large_blur, -0.5, 0)
        
        # 3. Strong Detail Enhancement (High Frequency)
        # sigma_s=10, sigma_r=0.15 gives a very 'HDR' look which sharpens textures
        enhanced = cv2.detailEnhance(large_details, sigma_s=10, sigma_r=0.15)
        
        # 4. Additional small unsharp mask on top for edges
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # 5. Convert back to Tensor
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        enhanced = enhanced.astype(np.float32) / 255.0
        enhanced_t = torch.from_numpy(enhanced).permute(2, 0, 1).unsqueeze(0).to(x.device)
        
        return torch.clamp(enhanced_t, 0, 1)

    def _forward_net(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        # Neck
        bot = self.neck(x3)
        
        # Decoder
        up1 = self.up1(bot)
        # Handle padding issues if size is odd
        if up1.size(2) != x2.size(2) or up1.size(3) != x2.size(3):
            up1 = F.interpolate(up1, size=(x2.size(2), x2.size(3)))
            
        cat1 = torch.cat([up1, x2], dim=1)
        d1 = self.dec1(cat1)
        
        up2 = self.up2(d1)
        if up2.size(2) != x1.size(2) or up2.size(3) != x1.size(3):
            up2 = F.interpolate(up2, size=(x1.size(2), x1.size(3)))
            
        cat2 = torch.cat([up2, x1], dim=1)
        d2 = self.dec2(cat2)
        
        out = self.final(d2)
        
        # Residual learning (predict residual and add to input) is standard for deblurring
        # out = x + out
        # But we'll just return sigmoid if we were training directly
        return torch.sigmoid(out)

    def load_weights(self, path):
        try:
            self.load_state_dict(torch.load(path))
            self.pretrained = True
        except Exception as e:
            print(f"Failed to load weights: {e}")
            self.pretrained = False
