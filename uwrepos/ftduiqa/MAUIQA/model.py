import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, 1, padding=0)

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        avg_out = F.relu(self.fc1(avg_out))
        avg_out = torch.sigmoid(self.fc2(avg_out))
        return x * avg_out

class MAUIQA(nn.Module):
    def __init__(self):
        super(MAUIQA, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7]) 
        
        self.attention1 = AttentionModule(256)
        self.attention2 = AttentionModule(512)
        self.attention3 = AttentionModule(1024)
        
        self.fc1 = nn.Linear(256 + 512 + 1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h1 = self.layer1(x) 
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        
        h1 = self.attention1(h1)
        h2 = self.attention2(h2)
        h3 = self.attention3(h3)
        
        h1 = F.adaptive_avg_pool2d(h1, (1, 1)).view(h1.size(0), -1)
        h2 = F.adaptive_avg_pool2d(h2, (1, 1)).view(h2.size(0), -1)
        h3 = F.adaptive_avg_pool2d(h3, (1, 1)).view(h3.size(0), -1)
        
        h = torch.cat([h1, h2, h3], dim=1)
        
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        
        q = self.fc3(h)
        return q

