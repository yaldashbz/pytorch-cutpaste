import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms

class ProjectionNet(nn.Module):
    def __init__(self, pretrained=True, head_layers=[512,512,512,512,512,512,512,512,128], num_classes=2):
        super(ProjectionNet, self).__init__()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm = transforms.Normalize(mean=mean, std=std)
        self.pretrained = pretrained
        self.resnet18 = resnet18(pretrained=pretrained)

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons
        
        #the last layer without activation

        head = nn.Sequential(
            *sequential_layers
          )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)
    
    def forward(self, x):
        # normalize
        if self.pretrained:
            x = self.norm(x)
        # x = F.normalize(x, dim=-1)
        
        embeds = self.resnet18(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return logits, embeds
    
    def freeze_resnet(self):
        # freez full resnet18
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        #unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True
            
    def unfreeze(self):
        #unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
