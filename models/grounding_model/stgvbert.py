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
from utils.misc import NestedTensor



logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}





def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}




try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.MODEL.HIDDEN_SIZE, config.MODEL.INTERMEDIATE_SIZE)
        if isinstance(config.MODEL.HIDDEN_ACT, str) or (
            sys.version_info[0] == 2 and isinstance(config.MODEL.HIDDEN_ACT, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.MODEL.HIDDEN_ACT]
        else:
            self.intermediate_act_fn = config.MODEL.HIDDEN_ACT

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.MODEL.INTERMEDIATE_SIZE, config.MODEL.HIDDEN_SIZE)
        self.LayerNorm = BertLayerNorm(config.MODEL.HIDDEN_SIZE, eps=1e-12)
        self.dropout = nn.Dropout(config.MODEL.HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertImageIntermediate(nn.Module):
    def __init__(self, config):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(config.MODEL.V_HIDDEN_SIZE, config.MODEL.V_INTERMEDIATE_SIZE)
        if isinstance(config.MODEL.V_HIDDEN_ACT, str) or (
            sys.version_info[0] == 2 and isinstance(config.MODEL.V_HIDDEN_ACT, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.MODEL.V_HIDDEN_ACT]
        else:
            self.intermediate_act_fn = config.MODEL.V_HIDDEN_ACT

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):
    def __init__(self, config):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(config.MODEL.V_INTERMEDIATE_SIZE, config.MODEL.V_HIDDEN_SIZE)
        self.LayerNorm = BertLayerNorm(config.MODEL.V_HIDDEN_SIZE, eps=1e-12)
        self.dropout = nn.Dropout(config.MODEL.V_HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertBiAttention(nn.Module):
    def __init__(self, config):
        super(BertBiAttention, self).__init__()
        if config.MODEL.BI_INTERMEDIATE_SIZE % config.MODEL.BI_NUM_ATTENTION_HEADS != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.MODEL.BI_INTERMEDIATE_SIZE, config.MODEL.BI_NUM_ATTENTION_HEADS)
            )

        self.visualization = config.MODEL.VISUALIZATION
        self.num_attention_heads = config.MODEL.BI_NUM_ATTENTION_HEADS
        self.attention_head_size = int(
            config.MODEL.BI_INTERMEDIATE_SIZE / config.MODEL.BI_NUM_ATTENTION_HEADS
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(config.MODEL.V_HIDDEN_SIZE, self.all_head_size)
        self.key1 = nn.Linear(config.MODEL.V_HIDDEN_SIZE, self.all_head_size)
        self.value1 = nn.Linear(config.MODEL.V_HIDDEN_SIZE, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(config.MODEL.V_ATTENTION_PROBS_DROPOUT_PROB)

        self.query2 = nn.Linear(config.MODEL.HIDDEN_SIZE, self.all_head_size)
        self.key2 = nn.Linear(config.MODEL.HIDDEN_SIZE, self.all_head_size)
        self.value2 = nn.Linear(config.MODEL.HIDDEN_SIZE, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(config.MODEL.ATTENTION_PROBS_DROPOUT_PROB)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1,#[batch,，m_box+1，m=768] [b,t,h,w,c]
        input_tensor2,#[batch,max_word_size+1，m=768]
        attention_mask2,#[batch,1，1，max_word_size+1]
    ):
        #for t sp and te input
        B, num_frames, H, W, C = input_tensor1.shape
        L = input_tensor2.shape[1]
        visual = input_tensor1.view(B, num_frames, H * W, C)

        # Spatial Pooling
        visual_sp = visual.mean(2)  # B, num_frames, C

        # Temporal Pooling
        visual_tp = visual.mean(1)  # B, H*W, C

        # Combination
        combined_visual = torch.cat([visual_sp, visual_tp],
                                    dim=1)  # B, num_frames+H*W, C


        # for vision input.
        mixed_query_layer1 = self.query1(combined_visual)
        mixed_key_layer1 = self.key1(combined_visual)
        mixed_value_layer1 = self.value1(combined_visual)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        # attention_scores1 = attention_scores1 + attention_mask1
        # if use_co_attention_mask:
        # attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        # if use_co_attention_mask:
        # attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        visual_sp, visual_tp = torch.split(context_layer2, [num_frames, H * W], dim=1)

        # Replication
        visual_sp = visual_sp.unsqueeze(2).expand(B, num_frames,
                                                  H * W, C).view(
            B, num_frames,
            H, W, C)
        visual_tp = visual_tp.unsqueeze(1).expand(B, num_frames,
                                                  H * W, C).view(
            B, num_frames,
            H, W, C)
        context_layer2 = visual_sp + visual_tp
        attn_data = None

        if self.visualization:
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer2,
                "keys1": key_layer1,
                "attn2": attention_probs2,
                "querues2": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data


class BertBiOutput(nn.Module):
    def __init__(self, config):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(config.MODEL.BI_HIDDEN_SIZE, config.MODEL.V_HIDDEN_SIZE)
        self.LayerNorm1 = BertLayerNorm(config.MODEL.V_HIDDEN_SIZE, eps=1e-12)
        self.dropout1 = nn.Dropout(config.MODEL.V_HIDDEN_DROPOUT_PROB)

        self.q_dense1 = nn.Linear(config.MODEL.BI_HIDDEN_SIZE, config.MODEL.V_HIDDEN_SIZE)
        self.q_dropout1 = nn.Dropout(config.MODEL.V_HIDDEN_DROPOUT_PROB)

        self.dense2 = nn.Linear(config.MODEL.BI_HIDDEN_SIZE, config.MODEL.HIDDEN_SIZE)
        self.LayerNorm2 = BertLayerNorm(config.MODEL.HIDDEN_SIZE, eps=1e-12)
        self.dropout2 = nn.Dropout(config.MODEL.HIDDEN_DROPOUT_PROB)

        self.q_dense2 = nn.Linear(config.MODEL.BI_HIDDEN_SIZE, config.MODEL.HIDDEN_SIZE)
        self.q_dropout2 = nn.Dropout(config.MODEL.HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):
    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention(config)

        self.biOutput = BertBiOutput(config)

        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)

        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(
        self,
        input_tensor1,
        input_tensor2,
        attention_mask2,

    ):

        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            input_tensor2,
            attention_mask2,
        )

        attention_output1, attention_output2 = self.biOutput(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class STVGBERT(nn.Module):
    def __init__(self, config):
        super(STVGBERT, self).__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.

        self.v_biattention_id = config.MODEL.V_BIATTENTION_ID
        connect_layer = BertConnectionLayer(config)

        self.c_layer = nn.ModuleList(
            [copy.deepcopy(connect_layer) for _ in range(len(self.v_biattention_id))]
        )

    def forward(
        self,
        txt_embedding,#[batch,max_word_size+1，m=768]
        image_embedding,#[batch,1,num_box+1,dim=1024]
        txt_attention_mask,#[batch,1,1,max_word_size+1]
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []

        use_co_attention_mask = False
        for v_layer_id in self.v_biattention_id:
           # do the bi attention.
            image_embedding, txt_embedding, co_attention_probs = self.c_layer[
                v_layer_id
            ](
                image_embedding,
                txt_embedding,
                txt_attention_mask,

            )

            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)


        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            all_encoder_layers_v,
            all_encoder_layers_t,
            (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c),
        )



# class SimpleClassifier(nn.Module):
#     def __init__(self, in_dim, hid_dim, out_dim, dropout):
#         super(SimpleClassifier, self).__init__()
#         layers = [
#             weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
#             nn.ReLU(),
#             nn.Dropout(dropout, inplace=True),
#             weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
#         ]
#         self.main = nn.Sequential(*layers)

#     def forward(self, x):
#         logits = self.main(x)
#         return logits
if __name__ == "__main__":
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
