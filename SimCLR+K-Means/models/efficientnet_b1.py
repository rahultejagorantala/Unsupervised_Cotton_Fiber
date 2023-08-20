"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch.nn as nn
import torchvision.models as models


def efficientnet_b1(train_last_only):
    # backbone = models.__dict__['efficientnet_b1']()
    backbone = models.efficientnet_b1(pretrained=False)
    backbone.classifier = nn.Identity()
    if train_last_only:
        for param in backbone.parameters():
            param.requires_grad = False
    return {'backbone': backbone, 'dim': 1280}