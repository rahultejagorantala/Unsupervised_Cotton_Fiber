"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch.nn as nn
import torchvision.models as models
from models.pretrained_resnet import Model

def resnet50(train_only_last = None, pretrained_resnet50 = None):
    if pretrained_resnet50:
        backbone = Model().f
        if train_only_last:
            for param in backbone.parameters():
                param.requires_grad = False
    else:
        backbone = models.resnet50(pretrained=False)
        backbone.fc = nn.Identity()
        if train_only_last:
            print("Training only last layers. froze all the remaining layers in the model")
            backbone = models.resnet50(pretrained=True)
            backbone.fc = nn.Identity()
            for param in backbone.parameters():
                param.requires_grad = False
    return {'backbone': backbone, 'dim': 2048}
