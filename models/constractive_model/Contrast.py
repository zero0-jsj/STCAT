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
from torch import nn

from models.constractive_model.Spatial_Contrast import Spatial_Contrast


class Contrast(nn.Module):
    def __init__(self, cfg):
        super(Contrast, self).__init__()
        self.sp_contrast =  Spatial_Contrast()





    def forward(self,videos, language, box_mask):
        return 1