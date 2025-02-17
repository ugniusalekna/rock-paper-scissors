import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.gelu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.gelu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)

        x = x.flatten(1)
        return self.fc(x)


class ImageClassifierV4(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 24, kernel_size=2, padding=0)
        
        self.conv1 = nn.Conv2d(24, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.gelu(self.conv0(x))

        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.bn2(self.conv2(x)))

        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.bn4(self.conv4(x)))

        x = F.gelu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.bn6(self.conv6(x)))

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.fc(x)