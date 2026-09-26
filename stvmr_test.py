import torch
from STVMR import STVMR
import random
from util import loss, nce_loss

B = 5     # batch_size大小
K = 9     # 最大查询数量
sen_shape = 784    # 查询特征长度

T = 64    # 视频帧数
N = 20    # 区域数量
video_shape = 4096   # 区域特征长度

instance_type = 'clip'    # 时间MIT时，instance的类型，有clip和frame两种
clip_length = [16, 24, 32, 48]    # 当instance类型为clip时，clip的长度列表

annotation_frame_index = 29   # 标记的帧的下标（从1开始）
sigma = 1.0     # 计算高斯分布值的超参数sigma
delta = 0.1     # 计算时间和空间损失值的时候的超参数delta
lamda = 0.9     # 计算最终损失值惩罚项的超参数lamda

feature_shape = 512   # 最终的查询和区域特征长度

# 模拟查询特征
sen_feature = torch.rand([B, K, sen_shape])
print(sen_feature.shape)

# 模拟查询mask(因为每一个句子的名词数量不同)
sen_mask = torch.zeros([B, K])
for i in range(B):
    for j in range(K):
        if random.random()>0.5:
            sen_mask[i][j] = 1

# 模拟区域特征
video_feature = torch.rand([B, T, N, video_shape])
print(video_feature.shape)

# 建立STVMR模型
stvmr = STVMR(sen_shape, video_shape, feature_shape, sigma, annotation_frame_index)

# 执行模型，通过查询特征和区域特征获得最终的时间得分和空间得分
spatio_similarity_socre, tem_similarity_score = stvmr(sen_feature, video_feature, instance_type, clip_length)

# 获取最终的loss值
loss = loss(spatio_similarity_socre, tem_similarity_score, sen_mask, delta, lamda)

nce_loss = nce_loss(spatio_similarity_socre, tem_similarity_score, sen_mask)

