import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
import torchvision
import torchvision.models.detection.rpn as rpn
import torchvision.models.detection.image_list as list
from torch import nn

from utils.comm import bbox_iou


class Spatial_Contrast(nn.Module):
    def __init__(self):
        super(Spatial_Contrast, self).__init__()
        self.roi_align = torchvision.ops.RoIAlign(output_size=3,spatial_scale=1,sampling_ratio=-1)
        self.fc1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1)
        self.anchor_generator = rpn.AnchorGenerator(sizes=((14,32,56,75),))
        self.img_shape = (224,224)

    def neg_sample(self,videos,features,pos):
        image_sizes = [self.img_shape] * 40
        image_list = list.ImageList(videos,image_sizes)
        anchors = self.anchor_generator(image_list,[features])
        n = anchors[0].shape[0]
        anchors = torch.cat(anchors,dim=0).view(40,n,4)
        anchors = torch.clamp(anchors,0,224)
        pos = pos.unsqueeze(1).expand(40,n,4)
        iou = bbox_iou(pos,anchors)
        num = torch.gt(iou,0.01).sum(axis=[1],keepdim=True)
        return anchors



    def forward(self,videos,feature,rois,pos):
        anchors = self.neg_sample(videos,feature,pos)
        x = self.roi_align(videos,[rois])
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('Test Layer')
    layer = Spatial_Contrast().to(device)

    B,  num_frames, H, W, C = 4, 10, 7, 7, 512
    pos = torch.cat([torch.Tensor([[46, 50, 110, 86]])] * B*num_frames,dim=0).to(device)
    videos = torch.randn(B*num_frames, C, 224, 224).to(device)
    visual = torch.randn(B*num_frames, C, H, W).to(device)
    rois = torch.rand(1, 4).to(device)
    visual_output = layer(videos,visual,rois,pos)
    print(visual_output.shape)

