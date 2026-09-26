# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from config import cfg

import pdb
from engine.evaluate import do_eval
from models.grounding_model.stgvbert import STVGBERT
from models.vision_model import build_vis_encoder



from models.language_model import build_text_encoder
from utils.misc import NestedTensor
from .grounding_model import build_decoder
from .net_utils import inverse_sigmoid, MLP


class DSCSTVG(nn.Module):
    def __init__(self, cfg):
        super(DSCSTVG, self).__init__()
        self.stvgbert = STVGBERT(cfg)
        self.B = cfg.SOLVER.BATCH_SIZE
        self.vis_encoder = build_vis_encoder(cfg)
        vis_fea_dim = self.vis_encoder.num_channels
        self.text_encoder = build_text_encoder(cfg)
        self.ground_decoder = build_decoder(cfg)
        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.
        self.max_video_len = cfg.INPUT.MAX_VIDEO_LEN
        self.use_attn = cfg.SOLVER.USE_ATTN

        self.use_aux_loss = cfg.SOLVER.USE_AUX_LOSS  # use the output of each transformer layer
        self.use_actioness = cfg.MODEL.STCAT.USE_ACTION
        self.query_dim = cfg.MODEL.STCAT.QUERY_DIM

        hidden_dim = cfg.MODEL.STCAT.HIDDEN
        self.use_ds = cfg.MODEL.USE_DSDECOMPOSE
        if self.use_ds:
            self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim//2, kernel_size=1)
        else:
            self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim , kernel_size=1)
        self.static = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=1)
        self.dynamcis = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=1)
        self.temp_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.softmax = nn.Softmax(dim=2)
        self.action_embed = None
        if self.use_actioness:
            self.action_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.3)

        # add the iteration anchor update
        self.ground_decoder.decoder.bbox_embed = self.bbox_embed

    def DSDecompose(self,vis_features):
        B, num_frames, C, H, W = vis_features.shape
        static_feature = self.static(vis_features.mean(1))
        dynamics_feature = self.dynamcis((vis_features[:,1:num_frames,:,:,:] - vis_features[:,0:num_frames-1,:,:,:]).view(B*(num_frames-1), C, H, W))
        ori_vis_features = vis_features[:,1:num_frames,:,:,:]
        static_map = static_feature.mean(1)
        dynamics_map = dynamics_feature.mean(1).view(B,num_frames-1,H,W)
        static_feature = torch.mul(static_map.unsqueeze(1).unsqueeze(1),ori_vis_features)
        dynamics_feature = torch.mul(dynamics_map.unsqueeze(2), ori_vis_features)
        return torch.cat([static_feature,dynamics_feature],dim=2)

    def forward(self,videos, #NestedTensor:mask[BxT,h,w],tensor[BxT,c,h,w]
                language #list:[sentence,...]
                ):
        # print("Test resnet101 model")
        B = self.B
        num_frames = videos.tensors.shape[0] // B
        vis_outputs, vis_pos_embed = self.vis_encoder(videos)
        vis_features, visual_mask, durations = vis_outputs.decompose()
        # vis_features =[BxT,C=256,H,W]
        vis_features = self.input_proj(vis_features)
        _, C, H, W = vis_features.shape

        if self.use_ds:
            vis_features = vis_features.view(B, num_frames, C, H, W)  # [BxT,C,H,W]->[B,T,C,H,W]
            vis_features = self.DSDecompose(vis_features)  # [B,T,C,H,W]->[B,t,Cx2,H,W]
            num_frames = num_frames - 1
            C = C * 2
            visual_mask = visual_mask.view(B, num_frames + 1, H, W)[:, 1:num_frames + 1, :,
                          :].contiguous()  # [BxT,H,W]->[B,t,H,W]
            vis_pos_embed = vis_pos_embed.view(B, num_frames + 1, C, H, W)[:, 1:num_frames + 1, :, :,
                            :].contiguous()  # [BxT,C,H,W]->[B,t,C,H,W]
            vis_pos_embed = vis_pos_embed.view(B * num_frames, C, H, W)  # [B,t,C,H,W]->[Bxt,C,H,W]
            durations = [d - 1 for d in durations]
        visual = vis_features.view(B, num_frames, C, H, W).transpose(2, 3).transpose(3, 4)  # [B,t,C,H,W]->[B,t,H,W,C]
        visual_mask = visual_mask.view(B * num_frames, H * W)  # [Bxt,HxW]



        # print("Test robert model")
        # vis_outputs=NestedTensor:mask[BxT,H,W],tensor[BxT,C=2048,H,W];
        # vis_pos_embed=[BxT,C=512,H,W]

        device = vis_features.device
        # text=[L,B,C]
        # text_mask=[B,L]
        texts, _ = self.text_encoder(language, device)
        text = texts[1].transpose(0, 1)
        text_mask = texts[0].unsqueeze(1).unsqueeze(2)
        # print(text_mask.shape)
        # print(text.shape)

        # print('Test STVGBERT Model')
        # visual_output:[B,t,H,W,C]
        # text_output:[B,L,C]
        visual_output, text_output, all_attention_mask = self.stvgbert(text, visual, text_mask)
        visual_output, text_output = visual_output[-1], text_output[-1]
        # print(visual_output.shape)
        # print(text_output.shape)
        visual_output = visual_output.view(B * num_frames, H * W, C)
        #
        # print('gen decoder input')
        text_fea_list = []
        text_memory_resized = text_output.transpose(0, 1)
        for i_b in range(B):
            frame_length = num_frames
            text_fea_list.append(
                torch.stack([text_memory_resized[:, i_b] for _ in range(frame_length)], dim=1)
            )
        text_output = torch.cat(text_fea_list, dim=1)  # [B,L,C]->[L,Bxt,C]
        encoded_memory = torch.cat([visual_output.transpose(0, 1), text_output],
                                   dim=0)  # [L,Bxt,C]+[HxW,Bxt,C]->[(L+HxW),Bxt,C]
        # print(encoded_memory.shape)

        text_attention_mask = text_mask.squeeze(1).squeeze(1)
        text_mask_list = []
        for i_b in range(B):
            frame_length = num_frames
            text_mask_list.append(
                torch.stack([text_attention_mask[i_b] for _ in range(frame_length)])
            )
        text_attention_mask = torch.cat(text_mask_list)  # [B,L]->[Bxt,L]
        mask = torch.cat([visual_mask, text_attention_mask], dim=1).bool()  # [Bxt,(L+HxW)]
        # print(mask.shape)

        frame_cls = visual_output.mean(1).squeeze(1)  # [Bxt,C]
        # print(frame_cls.shape)

        video_cls = visual_output.view(B, num_frames, H * W, C).view(B, num_frames * H * W, C).mean(1).squeeze(
            1)  # [1,C]
        # print(video_cls.shape)

        memory_cache = {'encoded_memory': encoded_memory, 'mask': mask, 'frames_cls': frame_cls,
                        'videos_cls': video_cls, 'durations': durations, 'fea_map_size': (H, W)}

        # print('Test decoder Model')
        # outputs=[num_layer,B,t,C],[num_layer,B,t,4]
        # outputs_temp=[num_layer,B,t,C],[num_layer,B,t,t]
        outputs, outputs_temp = self.ground_decoder(
            memory_cache=memory_cache, vis_pos=vis_pos_embed,
            text_cls=None
        )

        # print("Test predice head")
        out = {}
        if self.use_attn:
            time_hs, weights = outputs_temp
            out["weights"] = weights[-1]
        #
        # # the final decoder embeddings and the refer anchors
        hs, reference = outputs  # hs : [num_layers, b, T, d_model], reference : [num_layers, b, T, 4]
        # ###############  predict bounding box ################
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.bbox_embed(hs)
        tmp[..., :self.query_dim] += reference_before_sigmoid
        outputs_coord = tmp.sigmoid()  # [num_layers, b, T, 4]
        outputs_coord = outputs_coord.flatten(1, 2)
        out.update({"pred_boxes": outputs_coord[-1]})
        # #######################################################
        #
        # ###############  predict the start and end probability ################
        outputs_temp = self.temp_embed(time_hs)
        out.update({"pred_sted": outputs_temp[-1]})
        # #######################################################
        #
        if self.use_actioness:
            outputs_actioness = self.action_embed(time_hs)
            out.update({"pred_actioness": outputs_actioness[-1]})

        if self.use_aux_loss:
            out["aux_outputs"] = [
                {
                    "pred_sted": a,
                    "pred_boxes": b,
                }
                for a, b in zip(outputs_temp[:-1], outputs_coord[:-1])
            ]
            for i_aux in range(len(out["aux_outputs"])):
                if self.use_attn:
                    out["aux_outputs"][i_aux]["weights"] = weights[i_aux]
                if self.use_actioness:
                    out["aux_outputs"][i_aux]["pred_actioness"] = outputs_actioness[i_aux]

        return out


logger = logging.getLogger(__name__)

