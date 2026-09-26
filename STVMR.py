import torch.nn as nn
import torch
from util import Gaussian, get_clip_feature, normalize, get_box_acc, show_box
from config import cfg

from transformers import AutoModel, AutoTokenizer
from utils.box_utils import box_xyxy_to_cxcywh

class STVMR(nn.Module):

    def __init__(self, sen_shape, video_shape, feature_shape, bert_path):
        super().__init__()
        # 作用于查询的线性层
        self.sen_FC = nn.Linear(sen_shape, feature_shape)
        # 查询和区域的线性层的激活函数都是tanh
        self.tanh = nn.Tanh()
        # 作用于区域的线性层
        self.video_FC = nn.Linear(video_shape, feature_shape)
        # 超参数sigma
        self.sigma = cfg.STVMR.SIGMA
        # 获取corenlp预训练模型，用于提取名词
        # 加载并冻结bert模型参数
        self.bert = AutoModel.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def get_bert_feature(self, nlp, sentences):

        # print(sentences)
        # print(sentences)
        if not isinstance(sentences, list):
            sentences = [sentences]

        inputs = self.tokenizer(sentences, max_length=cfg.BERT.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to('cuda:0')
        token_type_ids = inputs['token_type_ids'].to('cuda:0')
        attention_mask = inputs['attention_mask'].to('cuda:0')
        # print(input_ids.shape)
        # print(token_type_ids.shape)

        outputs = self.bert(input_ids, attention_mask, token_type_ids)

        # 查询特征（B,K,D）这里的D为784
        outputs = outputs['last_hidden_state']
        # print("sen_out: ", outputs)

        # 获取名词的mask，大小为（B,K），为1表示该处的特征为名词的特征
        noun_word_index_list = []
        for sentence in sentences:
            sen_tag = nlp.pos_tag(sentence)  # 词性标注，其中'NN'表示名词

            noun_word_index = []  # 保存名词
            noun_word_index.extend([0]*token_type_ids.shape[-1])
            for i in range(len(sen_tag)):
                # 判断是否为名词
                if sen_tag[i][1] == 'NN':
                    noun_word_index[i+1] = 1
            noun_word_index_list.append(noun_word_index)

        return outputs, noun_word_index_list

    def do_eval(self, videos, sen_feature, frccn_feature, noun_word_mask, rois, device, target_roi, real_target_rois):

        # print(sen_feature.shape)
        # 首先将查询特征通过线性层变成特定维度（B，K，D）
        sentence_feature = self.sen_FC(sen_feature)
        sentence_feature = self.tanh(sentence_feature)

        # 将区域特征通过线性层转为特定维度（B，T，N，D）T为视频长度，或者clip数量
        region_feature = self.video_FC(frccn_feature)
        region_feature = self.tanh(region_feature)

        # print("sen: ", sentence_feature.shape)
        # print("region: ", region_feature.shape)
        # print("rois: ", rois.shape)
        # print("noun: ", noun_word_mask.shape)
        region_feature = region_feature.reshape(cfg.SOLVER.BATCH_SIZE, cfg.INPUT.TRAIN_SAMPLE_NUM, -1, cfg.STVMR.FEATURE_SHAPE)
        # print("region: ", region_feature.shape)
        rois = rois.reshape(cfg.SOLVER.BATCH_SIZE, cfg.INPUT.TRAIN_SAMPLE_NUM, -1, 5)
        # print("rois: ", rois.shape)

        # 转换为（B,T,D,N）
        region_feature = region_feature.transpose(2, 3)
        # print("region: ", region_feature.shape)
        # 再转换为（T,B,D,N）
        region_feature = region_feature.transpose(0, 1)
        # print("region: ", region_feature.shape)
        # 增加维度，变为（T,B,1,D,N）
        region_feature = region_feature.unsqueeze(2)
        # print("region: ", region_feature.shape)

        # 计算相似度（B，K，D） dot （T,B,1,D,N） = （T,B,B,K,N）
        similarity = torch.matmul(sentence_feature.unsqueeze(0).unsqueeze(0), region_feature)

        # print("simi: ", similarity.shape)
        # 转换维度（B,B,T,K,N）
        similarity = similarity.transpose(0, 2)
        # print("simi: ", similarity.shape)

        # 在区域维度，获取最大相似度（B，B，T，K）
        max_similarity, _ = similarity.max(-1)
        # print("max_simi: ", max_similarity.shape)

        # 转换维度（B，B，K，T）
        max_similarity = max_similarity.transpose(2, 3)
        # print("max_simi: ", max_similarity.shape)

        # max_similarity = max_similarity.squeeze()
        # print(max_similarity.shape)

        min_tem_score = max_similarity.min(-1)[0]
        # print("min_simi: ", min_tem_score.shape)
        min_tem_score = min_tem_score.unsqueeze(-1)
        # print("min_simi: ", min_tem_score.shape)

        max_tem_socre = max_similarity.max(-1)[0]
        max_tem_socre = max_tem_socre.unsqueeze(-1)
        # print("max_simi: ", max_tem_socre.shape)

        context_similarity_score = 1.0 * (max_similarity - min_tem_score) / (max_tem_socre - min_tem_score)
        # print("con: ", context_similarity_score.shape)

        noun_word_mask = noun_word_mask.unsqueeze(0).unsqueeze(0)
        # B, B, K, T -> B, T, B, K
        context_similarity_score = context_similarity_score.permute(0, 3, 1, 2)
        # print("noun: ", noun_word_mask.shape)
        context_score_mask = context_similarity_score * noun_word_mask
        # print("con: ", context_score_mask.shape)
        # B, T, B
        context_similarity_score = torch.sum(context_score_mask, -1) / torch.sum(noun_word_mask, -1)
        # print("con: ", context_similarity_score.shape)

        # B, B, T
        context_similarity_score = context_similarity_score.permute(0, 2, 1)
        similarity_mask = torch.eye(context_similarity_score.shape[0]).to('cuda:0').unsqueeze(-1)
        # print("con: ", context_similarity_score.shape)
        # print("mask: ", similarity_mask.shape)
        context_similarity_score = context_similarity_score * similarity_mask
        context_similarity_score = context_similarity_score.sum(0)
        # print("con: ", context_similarity_score.shape)

        pred_sted = torch.zeros(context_similarity_score.shape[0], context_similarity_score.shape[1], 2)
        pred_box = torch.zeros(context_similarity_score.shape[0], context_similarity_score.shape[1],  4)

        video_t = context_similarity_score.shape[1]
        batch_size = context_similarity_score.shape[0]

        for b in range(batch_size):
            for i in range(video_t):
                if context_similarity_score[b][i] >= cfg.STVMR.THRESHOLD:
                        pred_sted[b][i][0] = 1
                        break
            for i in range(video_t):
                if context_similarity_score[b][video_t-1-i] >= cfg.STVMR.THRESHOLD:
                        pred_sted[b][video_t-1-i][1] = 1
                        break

        # print(pred_sted)
        # B, B, T, K, N
        # print("sim: ", similarity.shape)
        # similarity = similarity.squeeze()
        similarity = similarity.permute(0, 2, 4, 1, 3)
        # B, T, N, B, K
        # print("sim: ", similarity.shape)
        # B, K
        # print("noun: ", noun_word_mask.shape)
        similarity = torch.sum(similarity*noun_word_mask, -1) / torch.sum(noun_word_mask, -1)
        # B, T, N, B
        # print("sim: ", similarity.shape)
        similarity = similarity.permute(0, 3, 1, 2)
        # B, B, T, N
        # print("sim: ", similarity.shape)
        similarity_mask = similarity_mask.unsqueeze(-1)
        # B, B, 1, 1
        # print("mask: ", similarity_mask.shape)
        similarity = (similarity * similarity_mask).sum(0)
        # B, T, N
        # print("sim: ", similarity.shape)
        _, indicate = torch.sort(similarity, dim=-1, descending=True)
        # B, T, N
        # print("in: ", indicate.shape)
        for b in range(batch_size):
            for i in range(video_t):
                pred_box[b][i] = rois[b, i, indicate[b][i][0], 1:]

        # print(pred_box)
        # print(rois)

        pred_box = pred_box.reshape(-1, 4)

        box_acc = get_box_acc(pred_box, target_roi)

        show_box(videos, real_target_rois, pred_box, target_roi)

        video_size = videos.shape[-1]
        # print(video_size)

        pred_box = pred_box / video_size
        # print(pred_box)
        pred_box = box_xyxy_to_cxcywh(pred_box)

        # print(pred_sted)

        pred_sted = pred_sted.to(device)
        pred_box = pred_box.to(device)

        outputs = {}

        outputs["pred_sted"] = pred_sted
        outputs["pred_boxes"] = pred_box
        outputs['box_acc'] = box_acc

        return outputs

        # print(video.shape)

    # 查询特征，区域特征，instance类型，clip的长度
    def forward(self, sen_feature, video_feature, annotation_frame_index):

        # 标记的帧
        self.mu = annotation_frame_index
        # print(sen_feature)
        # 首先将查询特征通过线性层变成特定维度（B，K，D）
        sentence_feature = self.sen_FC(sen_feature)
        sentence_feature = self.tanh(sentence_feature)
        sentence_feature = normalize(sentence_feature)
        # print("sen: ", sentence_feature)

        if cfg.STVMR.INSTANCE_TYPE == 'clip':
            spatio_feature = video_feature.clone().detach()
            # print(spatio_feature.shape)

        # 判断instance类型是否为clip，如果是，则需要计算clip级别的区域特征
        if cfg.STVMR.INSTANCE_TYPE == 'clip':
            video_feature, clip_information, clip_mask = get_clip_feature(video_feature, cfg.STVMR.CLIP_LENGTH, self.mu)
        else:
            clip_information = []
            clip_mask = torch.ones([video_feature.shape[0], video_feature.shape[1]]).to('cuda:0')

        # 将区域特征通过线性层转为特定维度（B，T，N，D）T为视频长度，或者clip数量
        region_feature = self.video_FC(video_feature)
        region_feature = self.tanh(region_feature)
        region_feature = normalize(region_feature)
        # print(region_feature)

        if cfg.STVMR.INSTANCE_TYPE == 'clip':
            spatio_region_feature = self.video_FC(spatio_feature)
            spatio_region_feature = self.tanh(spatio_region_feature)
            # 转换为（B,T,D,N）
            spatio_region_feature = spatio_region_feature.transpose(2, 3)
            # print(region_feature.shape)
            # 再转换为（T,B,D,N）
            spatio_region_feature = spatio_region_feature.transpose(0, 1)
            # print(region_feature.shape)
            # 增加维度，变为（T,B,1,D,N）
            spatio_region_feature = spatio_region_feature.unsqueeze(2)
            # print(region_feature.shape)

            # 计算相似度（B，K，D） dot （T,B,1,D,N） = （T,B,B,K,N）
            spatio_similarity = torch.matmul(sentence_feature, spatio_region_feature)
            # print(similarity.shape)
            # 转换维度（B,B,T,K,N）
            spatio_similarity = spatio_similarity.transpose(0, 2)
            spatio_max_similarity, _ = spatio_similarity.max(-1)
            # 转换维度（B，B，K，T）
            spatio_max_similarity = spatio_max_similarity.transpose(2, 3)
            # print("spatio: ",spatio_max_similarity.shape)
            spatio_max_similarity = spatio_max_similarity.permute(0, 2, 1, 3)
            # print(spatio_max_similarity.shape)

        # print(sentence_feature.shape)
        # print(region_feature.shape)

        # 转换为（B,T,D,N）
        region_feature = region_feature.transpose(2, 3)
        # print(region_feature.shape)
        # 再转换为（T,B,D,N）
        region_feature = region_feature.transpose(0, 1)
        # print(region_feature.shape)
        # 增加维度，变为（T,B,1,D,N）
        region_feature = region_feature.unsqueeze(2)
        # print(region_feature.shape)


        # 计算相似度（B，K，D） dot （T,B,1,D,N） = （T,B,B,K,N）
        similarity = torch.matmul(sentence_feature.unsqueeze(0).unsqueeze(0), region_feature)
        # print(similarity)
        # 转换维度（B,B,T,K,N）
        similarity = similarity.transpose(0, 2)
        # print(similarity.shape)

        # 在区域维度，获取最大相似度（B，B，T，K）
        max_similarity, _ = similarity.max(-1)
        # print(max_similarity)

        # 转换维度（B，B，K，T）
        max_similarity = max_similarity.transpose(2, 3)
        # print(max_similarity.shape)

        # 获取视频长度，或者是clip数量（T）
        video_length = max_similarity.shape[3]
        # print(video_length)

        if not cfg.STVMR.IS_GAUSS_LOSS:
            # 计算高斯值(B, T)
            gaussian_w = Gaussian(video_length, self.sigma, self.mu, cfg.STVMR.INSTANCE_TYPE, clip_information)
            # print("gauss: ", gaussian_w)

        # (B, K, B, T)
        max_similarity = max_similarity.permute(0, 2, 1, 3)
        # print(max_similarity.shape)

        if cfg.STVMR.INSTANCE_TYPE == 'clip':
            # print("spatio: ", spatio_max_similarity.shape)
            # 空间维度的最大相似度（B，B，K），获取标注帧的最大相似度
            spation_max_similarity = torch.zeros([spatio_max_similarity.shape[0], spatio_max_similarity.shape[1], spatio_max_similarity.shape[2]]).to('cuda:0')
            for i in range(self.mu.shape[0]):
                spation_max_similarity[:, :, i] = spatio_max_similarity[:, :, i, self.mu[i]-1]
            # print("final spatio:", spation_max_similarity.shape)
        else:
            # 空间维度的最大相似度（B，B，K），获取标注帧的最大相似度
            spation_max_similarity = torch.zeros(
                [max_similarity.shape[0], max_similarity.shape[1], max_similarity.shape[2]]).to('cuda:0')
            for i in range(self.mu.shape[0]):
                spation_max_similarity[:, :, i] = max_similarity[:, :, i, self.mu[i]-1]
            # B, k, B, T;  B
            # spation_max_similarity = max_similarity[..., torch.LongTensor(self.mu - 1)]
            # print(spation_max_similarity.shape)

        if not cfg.STVMR.IS_GAUSS_LOSS:
            # 时间维度的最大相似度需要乘以高斯权重，之后取平均（B，K，B）
            # B, k, B, T
            tem_max_similarity = max_similarity * gaussian_w
            # print(tem_max_similarity)
            # print("clip: ", clip_mask.shape)
            tem_max_similarity = torch.sum(tem_max_similarity*clip_mask, 3) / torch.sum(clip_mask, 1)
            # print("final tem: ", tem_max_similarity)
            # B, B, K
            tem_max_similarity = tem_max_similarity.permute(0, 2, 1)
        else:
            tem_max_similarity = max_similarity

        spation_max_similarity = spation_max_similarity.permute(0, 2, 1)

        if cfg.STVMR.IS_GAUSS_LOSS:
            return spation_max_similarity, tem_max_similarity, video_length, self.sigma, self.mu, clip_information

        # 返回时间维度和空间维度的最大相似度（即得分）
        return spation_max_similarity, tem_max_similarity