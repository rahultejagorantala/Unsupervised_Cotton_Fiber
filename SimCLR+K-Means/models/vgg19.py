"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch.nn as nn
import torchvision.models as models


def vgg19():
    backbone = models.__dict__['vgg19']()
    backbone.classifier = nn.Identity()
    return {'backbone': backbone, 'dim': 25088}