import os
import errno
import random
import torch
import numpy as np
import subprocess
from .comm import is_main_process


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_seed(seed):
    print("set seed ",seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


def to_device(targets, device):
    transfer_keys = set(['actioness', 'start_heatmap', 'end_heatmap', 'boxs', 'iou_map', 'candidates'])
    for idx in range(len(targets)):
        for key in targets[idx].keys():
            if key in transfer_keys:
                targets[idx][key] = targets[idx][key].to(device)
    return targets


class NestedTensor(object):
    def __init__(self, tensors, mask, durations, box_mask):
        self.tensors = tensors
        self.mask = mask
        self.durations = durations
        self.box_mask = box_mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_box_mask = self.box_mask.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask, self.durations, cast_box_mask)

    def do_DSDecompose(self):
        T = self.durations[0]
        _, c, h, w = self.tensors.shape
        self.tensors = self.tensors.view(self.tensors.shape[0] // T, T, c, h, w)
        self.tensors = self.tensors[:, 1:, :, :, :].contiguous().view(-1, c, h, w)
        dura = [x - 1 for x in self.durations]
        self.durations = dura
        self.mask = self.mask.view(self.mask.shape[0] // T, T, h, w)
        self.mask = self.mask[:, 1:, :, :].contiguous().view(-1, h, w)

    def decompose(self):
        return self.tensors, self.mask, self.durations

    def subsample(self, stride, start_idx=0):
        # Subsample the video for multi-modal Interaction
        sampled_tensors = [video[start_idx::stride] for video in \
                           torch.split(self.tensors, self.durations, dim=0)]
        sampled_mask = [mask[start_idx::stride] for mask in \
                        torch.split(self.mask, self.durations, dim=0)]

        sampled_durations = [tensor.shape[0] for tensor in sampled_tensors]

        return NestedTensor(torch.cat(sampled_tensors, dim=0),
                            torch.cat(sampled_mask, dim=0), sampled_durations)

    @classmethod
    def from_tensor_list(cls, tensor_list, targets):
        assert tensor_list[0].ndim == 4  # videos
        T, c, h, w = tensor_list[0].shape
        B = len(tensor_list)
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        # total number of frames in the batch
        durations = [tensor.shape[0] for tensor in tensor_list]
        tensor = torch.zeros((B * T, c, h, w), dtype=dtype, device=device)
        mask = torch.ones((B * T, h, w), dtype=torch.bool, device=device)
        if (len(targets[0]["frame_ids"]) != T):
            box_mask = torch.zeros((B, T - 1, h, w), dtype=torch.bool, device=device)
        else:
            box_mask = torch.zeros((B, T, h, w), dtype=torch.bool, device=device)
        cur_dur = 0
        for i_clip, (clip, target) in enumerate(zip(tensor_list, targets)):
            tensor[
            cur_dur: cur_dur + T,
            : c,
            : h,
            : w,
            ].copy_(clip)
            mask[
            cur_dur: cur_dur + T, : h, : w
            ] = False

            for i, act in enumerate(target['actioness']):
                if(act > 0):
                    box_mask[i_clip,
                    i,
                    int(target['boxs'].bbox[i-target['start_idx']][1]):int(target['boxs'].bbox[i-target['start_idx']][3])+1,
                    int(target['boxs'].bbox[i-target['start_idx']][0]):int(target['boxs'].bbox[i-target['start_idx']][2])+1
                    ]=True

            cur_dur += clip.shape[0]
        box_mask = box_mask.view(B*T,h,w)
        return cls(tensor, mask, durations,box_mask), box_mask

    def __repr__(self):
        return repr(self.tensors)