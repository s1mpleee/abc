import torch.nn as nn
import math

import torch
from efficientnet_pytorch import EfficientNet





class mmm(nn.Module):
    def __init__(self):
        super(mmm, self).__init__()
        #self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model = EfficientNet.from_name('efficientnet-b0')
        
       
        self.avg_pool = nn.AdaptiveMaxPool2d((1,1))
       
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1280, 2))

    def forward(self, x):
        x = self.model.extract_features(x)
        
       
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def build_net():
    return mmm()


