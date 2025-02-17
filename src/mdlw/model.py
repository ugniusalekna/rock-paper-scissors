import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(64)
        
        self.adapool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = F.gelu(self.bn5(self.conv5(x)))
        x = F.gelu(self.bn6(self.conv6(x)))
        
        x = self.adapool(x)
        x = x.flatten(1)

        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


# class ImageClassifier(nn.Module):
#     def __init__(self, num_classes=3):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.bn1   = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn2   = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn3   = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.adapool = nn.AdaptiveAvgPool2d((4, 4))
#         self.fc1 = nn.Linear(64 * 4 * 4, 256)
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)
        
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x)
        
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool(x)
        
#         x = self.adapool(x)
#         x = x.flatten(1)
        
#         x = F.relu(self.fc1(x))
#         logits = self.fc2(x)
        
#         return logits