import torch
import numpy as np
from config import cfg
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import cv2

# 获取高斯权重，视频长度（或者clip数量），超参数sigma，标注帧下标，instance类别，clip信息（初始帧，长度）
def Gaussian(video_length, sigma, mu, instance_type, clip_information):
    # 判断是否为clip级别，clip级别需要用每个clip最中间那帧的下标计算高斯分布值
    if instance_type == 'clip':
        gauss_tensor = torch.tensor([[((i-1) + l/2) for i,l in clip_information[k]] for k in range(len(clip_information))])
        # print("gauss: ",gauss_tensor.shape)
    else:
        # 帧下标从1到video_length(B*T)
        gauss_tensor = torch.arange(1, video_length+1)
        gauss_tensor = gauss_tensor.expand(mu.shape[0], video_length)
        # print(gauss_tensor.shape)
    # print(gauss_tensor)

    # 让值归一化到-1到1之间
    gauss_tensor = norm_fun(gauss_tensor, video_length)
    # 标注的那一帧的下标也要对应归一化到-1到1之间

    mu = norm_fun(mu, video_length)
    mu = mu.unsqueeze(1).expand(gauss_tensor.shape[0], gauss_tensor.shape[1]).cpu().detach().numpy()
    # print("mu: ", mu.shape)

    # print(gauss_tensor.shape)
    gauss_tensor = gauss_tensor.cpu().detach().numpy()

    # 计算高斯分布值
    sqrt_2pi = np.power(2*np.pi, 0.5)
    coef = 1.0 / (sqrt_2pi * sigma)
    powercoef = -1.0 / (2*np.power(sigma, 2))
    mypow = powercoef * (np.power((gauss_tensor - mu), 2))
    # 进行归一化，不然后面np.exp会越界
    # mypow = (mypow - np.mean(mypow)) / np.std(mypow)
    result = coef * (np.exp(mypow))
    # 将结果归一化到0-1之间
    # result = (result - np.min(result, axis=1, keepdims=True)) / (np.max(result, axis=1, keepdims=True) - np.min(result, axis=1, keepdims=True))
    result = result / np.sum(result, axis=1, keepdims=True)
    result = torch.tensor(result).to('cuda:0')
    # print(mu[0])
    # my_ploy(gauss_tensor[0], result[0].cpu().detach().numpy())

    return result

def my_ploy(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("t")
    plt.ylabel("guass")
    plt.legend()
    plt.show()

# 将自变量归一化到-1到1之间
def norm_fun(gauss_tensor, length):
    new_guass_tensor = (gauss_tensor-1)*2/(length-1)-1
    return new_guass_tensor

# 对clip维度的所有帧，做最大池化
def maxpool(inputs):
    return torch.max(inputs, 0)[0]

def get_top_20_region_feature(frccn_feature, anno_index, anno_f):
    # print(anno_f)
    # print(anno_index)

    anno_box_feature = frccn_feature[anno_f, anno_index]
    # print(anno_box_feature.shape)
    anno_box_feature = anno_box_feature.unsqueeze(1).unsqueeze(3)
    # print(anno_box_feature.shape)
    frccn_feature = frccn_feature.reshape(cfg.SOLVER.BATCH_SIZE, cfg.INPUT.TRAIN_SAMPLE_NUM, -1, cfg.STVMR.REGION_SHAPE)
    similaryity_region = torch.matmul(frccn_feature, anno_box_feature).squeeze()
    # print(similaryity_region.shape)
    _, indicate = torch.sort(similaryity_region, -1)
    # print(indicate.shape)
    indicate = indicate[:, :, :20]
    # print(indicate.shape)
    region_feature = torch.zeros(cfg.SOLVER.BATCH_SIZE, cfg.INPUT.TRAIN_SAMPLE_NUM, 20, cfg.STVMR.REGION_SHAPE)
    for i in range(frccn_feature.shape[0]):
        for j in range(frccn_feature.shape[1]):
            region_feature[i, j] = frccn_feature[i, j, indicate[i][j]]
    # print(region_feature.shape)

    return region_feature

# 获取clip级别的特征还有对应的信息（初始下标，长度）
def get_clip_feature(video_feature, clip_lengths, ano_frame):

    # 记录clip的初始下标和长度
    clip_information = torch.zeros([video_feature.shape[0], cfg.STVMR.MAX_CLIP_NUM, 2]).to('cuda:0')
    # 记录clip级别的特征
    final_video_feature = torch.zeros([video_feature.shape[0], cfg.STVMR.MAX_CLIP_NUM, video_feature.shape[2], video_feature.shape[3]]).to('cuda:0')

    clip_mask = torch.zeros([video_feature.shape[0], cfg.STVMR.MAX_CLIP_NUM]).to('cuda:0')
    # print(video_feature.shape)

    for j in range(ano_frame.shape[0]):
        k = 0
        # 遍历所有clip长度
        for len_v in clip_lengths:
            # 以2为间隔，遍历所有可能的初始下标
            for i in range(max(1, ano_frame[j]-len_v+1), min(cfg.INPUT.TRAIN_SAMPLE_NUM, ano_frame[j]+len_v-1), 2):
                if k >= cfg.STVMR.MAX_CLIP_NUM:
                    continue
                # 对clip中的所有帧进行最大池化
                final_video_feature[j][k] = maxpool(video_feature[j,i-1:i+len_v-1,:])
                # print(clip_video_feature.shape)
                # 记录信息
                clip_information[j][k] = torch.tensor([i, len_v])

                clip_mask[j][k] = 1
                k = k+1

    # 将clip信息转换为张量（B, T，2）
    # print("clip_information: ", clip_information.shape)
    # 将其转换为一整个张量（B,T,N,D）
    # print("clip")
    # print(clip_information.shape)
    # print(final_video_feature.shape)
    # print(clip_mask.shape)
    return final_video_feature, clip_information, clip_mask

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def get_box_acc(pred_box, target_box):
    target_box_index = target_box[:, 0].squeeze().type(torch.long).to("cuda:0")
    target_box = target_box[:, 1:].to("cuda:0")
    pred_box = pred_box.to("cuda:0")
    box_acc = 0
    for i in range(len(target_box)):
        if pred_box[target_box_index[i]].equal(target_box[i]):
            box_acc = box_acc + 1
    box_acc = box_acc / len(target_box)
    return box_acc

def show_box(videos, real_target_rois, pred_box, target_roi):
    # print(videos.shape)
    # print(rois.shape)
    # print(pred_box.shape)
    # print(target_roi.shape)
    image_path = cfg.OUTPUT_DIR + "/image"
    target_box_index = target_roi[:, 0].squeeze().type(torch.long).to("cuda:0")
    target_box = target_roi[:, 1:].to("cuda:0")
    pred_box = pred_box.to("cuda:0")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(len(target_box_index)):
        index = target_box_index[i]
        # print(index)
        array = videos[index].cpu().detach().numpy()
        for k in range(len(mean)):
            array[k] = array[k] * std[k] + mean[k]
        array = array * 255
        array = np.transpose(array, (1, 2, 0))
        array = array.astype(np.uint8)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        # print(array.shape)
        # print(array)
        im = Image.fromarray(array)
        draw = ImageDraw.Draw(im)
        bbox = real_target_rois[i, 1:]
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(255, 0, 0), width=2)
        bbox = pred_box[index]
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(0, 0, 255), width=2)
        bbox = target_box[i]
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(0, 255, 0), width=2)
        out_name = str(i) + ".png"
        im.save(os.path.join(image_path, out_name))
        # im.show()
        del draw


# 根据时间最大相似度和空间最大相似度计算最终的loss值
def loss(spatio_similarity_score, tem_similarity_score, sen_mask, delta_t, delta_s, lamda):

    # 获取batch_size
    batch_size = len(tem_similarity_score)
    # print(batch_size)

    similarity_mask = torch.eye(batch_size).to('cuda:0')
    # print(similarity_mask)
    # print(similarity_mask.shape)
    similarity_mask = similarity_mask.unsqueeze(-1)
    # print(similarity_mask.shape)
    # print(tem_similarity_score.shape)

    # max_tem_score, _ = tem_similarity_score.max(0)[0].max(0)
    # print(max_tem_score)
    # max_tem_score = max_tem_score.unsqueeze(0).unsqueeze(0)
    # print(max_tem_score.shape)
    # print(tem_similarity_score[:,:,0])

    # tem_similarity_score = tem_similarity_score - max_tem_score
    # print(tem_similarity_score)
    # print(tem_similarity_score[:,:,1])
    tem_similarity_score = tem_similarity_score / cfg.STVMR.TAU
    tem_similarity_score = torch.exp(tem_similarity_score)

    # max_spatio_score, _ = spatio_similarity_score.max(0)[0].max(0)
    # print(max_tem_score)
    # max_spatio_score = max_spatio_score.unsqueeze(0).unsqueeze(0)
    # print(max_tem_score.shape)
    # print(tem_similarity_score[:,:,0])

    # spatio_similarity_score = spatio_similarity_score - max_spatio_score
    spatio_similarity_score = spatio_similarity_score / cfg.STVMR.TAU
    # print(tem_similarity_score[:,:,0])
    # print(tem_similarity_score[:,:,1])
    spatio_similarity_score = torch.exp(spatio_similarity_score)

    tem_pos_score = tem_similarity_score * similarity_mask
    # print(tem_similarity_score.shape)
    # print(tem_pos_score.shape)
    tem_neg_score = tem_similarity_score - tem_pos_score

    spatio_pos_score = spatio_similarity_score * similarity_mask
    spatio_neg_score = spatio_similarity_score - spatio_pos_score

    tem_pos_score = tem_pos_score.sum(0)
    # tem_neg_score = tem_neg_score.sum(0).sum(0).unsqueeze(0)
    tem_neg_score = tem_neg_score.sum(0) + tem_neg_score.sum(1)
    tem_all_score = tem_pos_score + tem_neg_score
    spatio_pos_score = spatio_pos_score.sum(0)
    # spatio_neg_score = spatio_neg_score.sum(0).sum(0).unsqueeze(0)
    spatio_neg_score = spatio_neg_score.sum(0) + spatio_neg_score.sum(1)
    spatio_all_score = spatio_pos_score + spatio_neg_score
    # print("tem")
    # print(tem_pos_score)
    # print(tem_all_score)
    # print("spatio")
    # print(spatio_neg_score)
    # print(spatio_pos_score)

    loss_tem = -torch.log(tem_pos_score / tem_all_score)
    # print("loss: ", loss_tem)

    loss_spatio = -torch.log(spatio_pos_score / spatio_all_score)
    # print(loss_spatio)
    # loss_tem_list = []      # 记录时间loss
    # loss_spatio_list = []   # 记录空间loss
    # tem_pos_score_list = [] # 记录所有时间MIT正例的得分（用于计算损失度）
    # # tem_neg_score_list = []
    #
    # # 遍历所有batch
    # for i in range(batch_size):
    #     loss_tem = 0   # 时间loss
    #     loss_spatio = 0   # 空间loss
    #     # 时间正例和空间正例
    #     tem_pos_score = tem_similarity_score[i][i]
    #     spatio_pos_score = spatio_similarity_score[i][i]
    #     # print(tem_pos_score.shape)
    #
    #     # 为了计算惩罚项，需要将所有时间正例连接在一起
    #     tem_pos_score_list.append(tem_pos_score)
    #     # tem_neg_score_a_list = []
    #
    #     for j in range(batch_size):
    #         if i != j:
    #             # 计算时间MIT的损失
    #             # 时间负例（2，K）
    #             tem_neg_score = torch.stack([tem_similarity_score[i][j], tem_similarity_score[j][i]], 0)
    #             # print(tem_neg_score.shape)
    #             # 计算负例对的最大值（K）
    #             tem_neg_score, _ = torch.max(tem_neg_score, 0)
    #             # tem_neg_score_a_list.append(tem_neg_score)
    #             # print(tem_neg_score.shape)
    #             # 计算时间MIT的得分
    #             tem_score = tem_neg_score - tem_pos_score + delta_t
    #             # print(tem_score)
    #             # 与0取最大值，相当于做ReLU操作，将每一次负例的损失累加
    #             loss_tem = loss_tem + torch.nn.ReLU()(tem_score)
    #             # print(loss_tem)
    #             # 计算空间MIT的损失
    #             # 空间负例
    #             spatio_neg_score = torch.stack([spatio_similarity_score[i][j], spatio_similarity_score[j][i]], 0)
    #             # 计算负例对的最大值
    #             spatio_neg_score, _ = torch.max(spatio_neg_score, 0)
    #             # 计算空间MIT的得分
    #             spatio_score = spatio_neg_score - spatio_pos_score + delta_s
    #             # 与0取最大值，相当于做ReLU操作，将每一次负例的损失累加
    #             loss_spatio = loss_spatio + torch.nn.ReLU()(spatio_score)
    #
    #     # tem_neg_score_a_list = torch.stack(tem_neg_score_a_list)
    #     # print(tem_neg_score_a_list.shape)
    #     # tem_neg_score_a_list = torch.mean(tem_neg_score_a_list, 0)
    #     # print(tem_neg_score_a_list.shape)
    #     # tem_neg_score_list.append(tem_neg_score_a_list)
    #     # 负例对的数目为batch_size - 1，计算平均损失
    #     loss_tem = loss_tem / (batch_size - 1)
    #     loss_tem_list.append(loss_tem)
    #
    #     loss_spatio = loss_spatio / (batch_size - 1)
    #     loss_spatio_list.append(loss_spatio)
    #
    # # 将所有时间loss和空间loss都转换为张量（B,K）
    # loss_tem_final = torch.stack(loss_tem_list)
    # # print(loss_tem_final.shape)
    # loss_spatio_final = torch.stack(loss_spatio_list)
    # # print(loss_spatio_final.shape)
    #
    # # 所有时间正例，乘以-lamda就是惩罚项（B.K）
    # tem_pos_score_final = torch.stack(tem_pos_score_list)
    # # tem_neg_score_final = torch.stack(tem_neg_score_list)
    # # print("score")
    # # print(tem_pos_score_final)
    # # print(tem_neg_score_final)

    # 计算最后的总损失，为时间损失+空间损失+惩罚项（B，K）
    loss_final = loss_tem + loss_spatio
    # print("loss")
    # print(loss_tem)
    # print(loss_spatio)
    # print(loss_final)
    # print(sen_mask)

    # sum_sen_mask = torch.sum(sen_mask, 1)
    # loss_sum_list = torch.sum(loss_final*sen_mask, 1)
    loss = (loss_final * sen_mask).sum() / (sen_mask.sum() + 1e-10)
    # loss_sum = []
    # for i in range(len(sum_sen_mask)):
    #     if sum_sen_mask[i]!=0:
    #         loss_sum.append(loss_sum_list[i]/sum_sen_mask[i])


    # 计算最后的平均loss（每一个样本的名词数量不相同，需要乘以对应mask）（B，1）
    # print("final loss")
    # loss = torch.stack(loss_sum)
    # print(loss)
    # 对batch中所有样例的loss取平均
    # loss = torch.mean(loss_sum)
    # print(loss)
    return loss

# 计算gauss_nce_loss
def gauss_nce_loss(spatio_similarity_score, tem_similarity_score, sen_mask, video_length, sigma, mu, clip_information):

    # 获取batch_size
    batch_size = len(tem_similarity_score)
    # print(batch_size)

    similarity_mask = torch.eye(batch_size).to('cuda:0')
    # print(similarity_mask)
    # print(similarity_mask.shape)
    similarity_mask = similarity_mask.unsqueeze(-1).unsqueeze(-1)

    # B, K, B, T
    tem_similarity_score = tem_similarity_score / cfg.STVMR.TAU
    tem_similarity_score = torch.exp(tem_similarity_score)

    # B, B, K, T
    tem_similarity_score = tem_similarity_score.permute(0, 2, 1, 3)

    # spatio_similarity_score = spatio_similarity_score - max_spatio_score
    spatio_similarity_score = spatio_similarity_score / cfg.STVMR.TAU
    # print(tem_similarity_score[:,:,0])
    # print(tem_similarity_score[:,:,1])
    spatio_similarity_score = torch.exp(spatio_similarity_score)

    tem_pos_score = tem_similarity_score * similarity_mask
    # print(tem_similarity_score.shape)
    # print(tem_pos_score.shape)
    tem_neg_score = tem_similarity_score - tem_pos_score

    spatio_pos_score = spatio_similarity_score * similarity_mask
    spatio_neg_score = spatio_similarity_score - spatio_pos_score

    # B, K, T
    tem_pos_score = tem_pos_score.sum(0)
    # tem_neg_score = tem_neg_score.sum(0).sum(0).unsqueeze(0)
    tem_neg_score = tem_neg_score.sum(0) + tem_neg_score.sum(1)
    tem_all_score = tem_pos_score + tem_neg_score
    spatio_pos_score = spatio_pos_score.sum(0)
    # spatio_neg_score = spatio_neg_score.sum(0).sum(0).unsqueeze(0)
    spatio_neg_score = spatio_neg_score.sum(0) + spatio_neg_score.sum(1)
    spatio_all_score = spatio_pos_score + spatio_neg_score
    # print("tem")
    # print(tem_pos_score)
    # print(tem_all_score)
    # print("spatio")
    # print(spatio_neg_score)
    # print(spatio_pos_score)

    # 计算高斯值(B, T)
    gaussian_w = Gaussian(video_length, sigma, mu, cfg.STVMR.INSTANCE_TYPE, clip_information)
    # print("gauss: ", gaussian_w)

    # B, K, T
    loss_tem = -torch.log(tem_pos_score / tem_all_score)
    # print("loss: ", loss_tem)
    # K, B, T
    loss_tem = loss_tem.permute(1, 0, 2)
    loss_tem = loss_tem * gaussian_w
    # K, B
    loss_tem = loss_tem.sum(-1)
    loss_tem = loss_tem.permute(1, 0)

    loss_spatio = -torch.log(spatio_pos_score / spatio_all_score)

    # 计算最后的总损失，为时间损失+空间损失+惩罚项（B，K）
    loss_final = loss_tem + loss_spatio

    loss = (loss_final * sen_mask).sum() / (sen_mask.sum() + 1e-10)

    # print(loss)
    return loss

