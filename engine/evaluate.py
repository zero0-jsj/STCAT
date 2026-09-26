import torch
import torch.nn
from typing import Dict

from utils.misc import to_device
from utils.comm import synchronize, is_main_process
from tqdm import tqdm
from fasterrcnn.get_frcnn_feature import get_frcnn_feature, frcnn_load
from stanfordcorenlp import StanfordCoreNLP
import random


@torch.no_grad()
def linear_interp(bbox_dict):
    frame_ids = sorted([fid for fid in bbox_dict])
    if len(frame_ids) < 2:
        return bbox_dict
    for idx in range(0, len(frame_ids) - 1):
        left_fid = frame_ids[idx]
        right_fid = frame_ids[idx + 1]
        if right_fid - left_fid > 1:
            interval = right_fid - left_fid
            delta_x1 = (bbox_dict[right_fid][0][0] - bbox_dict[left_fid][0][0]) / interval
            delta_y1 = (bbox_dict[right_fid][0][1] - bbox_dict[left_fid][0][1]) / interval
            delta_x2 = (bbox_dict[right_fid][0][2] - bbox_dict[left_fid][0][2]) / interval
            delta_y2 = (bbox_dict[right_fid][0][3] - bbox_dict[left_fid][0][3]) / interval
            for step in range(1, interval):
                bbox_dict[left_fid + step] = [[
                  bbox_dict[left_fid][0][0] + step * delta_x1, 
                  bbox_dict[left_fid][0][1] + step * delta_y1, 
                  bbox_dict[left_fid][0][2] + step * delta_x2, 
                  bbox_dict[left_fid][0][3] + step * delta_y2, 
                ]]
    
    frame_ids = sorted([fid for fid in bbox_dict])
    assert max(frame_ids) - min(frame_ids) + 1 == len(frame_ids) 
    return {fid : bbox_dict[fid] for fid in frame_ids}


@torch.no_grad()
def single_forward(cfg, target_roi, real_target_rois, videos, texts, model, sen_feature, frccn_feature, noun_word_mask, rois, targets, device, postprocessor):

    outputs = model.do_eval(videos.tensors, sen_feature, frccn_feature, noun_word_mask, rois, device, target_roi, real_target_rois)
    box_acc = outputs['box_acc']
    # outputs = model(videos, texts)
    # print(outputs["pred_boxes"])
    # print(outputs["pred_sted"].shape, outputs["pred_boxes"].shape)
    # print(targets[0]['boxs'].bbox)
    if cfg.MODEL.USE_DSDECOMPOSE:
        videos.do_DSDecompose()
    durations = videos.durations
    b = len(durations)
    t = max(durations)
    batch_img_size = [list(target['ori_size']) for target in targets]
    orig_target_sizes = [img_size for img_size in batch_img_size for _ in range(t)]
    orig_target_sizes = torch.tensor(orig_target_sizes, device=device)
    # print("ori: ", orig_target_sizes.shape)
    assert orig_target_sizes.shape[0] == outputs['pred_boxes'].shape[0]

    frames_ids = [target['frame_ids'] for target in targets]
    pred_boxs, pred_steds = postprocessor(outputs, orig_target_sizes, frames_ids, durations)
    pred_boxs = pred_boxs.view(b, t, 4)

    vids = [target['item_id'] for target in targets]
    bbox_pred, temp_pred = {}, {}

    for i_b in range(b):
        frames_id = frames_ids[i_b]
        bbox_pred[vids[i_b]] = {}
        assert durations[i_b] == len(frames_id)
        for idx in range(durations[i_b]):
            bbox_pred[vids[i_b]][frames_id[idx]] = [pred_boxs[i_b][idx].detach().cpu().tolist()]

    if cfg.DATASET.NAME == 'VidSTG':
        qtypes = [target['qtype'] for target in targets]
        assert len(pred_steds) == len(qtypes)
        for i_b in range(b):
            temp_pred[vids[i_b]] = {
                "sted": pred_steds[i_b],
                "qtype": qtypes[i_b],
            }
    else:
        for i_b in range(b):
            temp_pred[vids[i_b]] = {
                "sted": pred_steds[i_b]
            }

    return bbox_pred, temp_pred, box_acc
    

@torch.no_grad()
def do_eval(cfg, mode, logger, model, postprocessor, data_loader, evaluator, device):
    """
    Video Spatial-Temporal Grounding Evaluation
    """
    model.eval()
    logger.info("Start evaluation on the {} split of {} dataset".format(mode, cfg.DATASET.NAME))

    fasterRCNN = frcnn_load()
    nlp = StanfordCoreNLP(cfg.BERT.CORENLP_PAYH)
    all_box_acc = 0
    batch_num = 0

    for i, batch_dict in enumerate(tqdm(data_loader)):
        print("batch_%d:" % (i))
        batch_num = i
        videos = batch_dict['videos'].to(device)
        texts = batch_dict['texts']
        targets = batch_dict['targets']
        # sen_feature = []
        # frccn_feature = []
        # noun_word_mask = []
        # rois = []

        video = videos.tensors
        # print(video.shape)
        # print(texts)
        text = texts
        real_target_rois = []
        for batch in range(len(targets)):
            actions = targets[batch]['actioness']
            boxes = targets[batch]['boxs'].bbox
            no_zero_inedx = torch.nonzero(actions)
            for j in range(len(boxes)):
                anno_box = boxes[j]
                # print(anno_box)
                image_size = video.shape[-1]
                # print(image_size)
                roi = torch.zeros([5])
                roi[1] = (anno_box[0] - anno_box[2] * 0.5)
                roi[2] = (anno_box[1] - anno_box[3] * 0.5)
                roi[3] = (anno_box[0] + anno_box[2] * 0.5)
                roi[4] = (anno_box[1] + anno_box[3] * 0.5)
                roi = roi * image_size
                roi[0] = j + no_zero_inedx[0][0] + batch * cfg.INPUT.TRAIN_SAMPLE_NUM
                # print(roi)
                real_target_rois.append(roi)
        real_target_rois = torch.stack(real_target_rois).to(device)
        # print(target_rois)
        frccn_feature, target_roi, rois = get_frcnn_feature(video, real_target_rois, fasterRCNN, False)
        sen_feature, noun_word_mask = model.get_bert_feature(nlp, text)
        sen_feature = sen_feature.to(device)
        noun_word_mask = torch.tensor(noun_word_mask)
        noun_word_mask = noun_word_mask.to(device)

        frccn_feature = frccn_feature.to(device)
        rois = rois.to(device)

        # print(videos.tensors.shape)
        bbox_pred, temp_pred, box_acc = single_forward(cfg, target_roi, real_target_rois, videos, texts, model, sen_feature, frccn_feature, noun_word_mask, rois,
                                              targets, device, postprocessor)
        all_box_acc = all_box_acc + box_acc
        # print(all_box_acc)
        # print("pred: ", bbox_pred)
        for vid in bbox_pred:
            interped_bbox_pred = linear_interp(bbox_pred[vid])
            bbox_pred[vid] = interped_bbox_pred

        evaluator.update(bbox_pred)
        evaluator.video_update(temp_pred)


    all_box_acc = all_box_acc / (batch_num + 1)
    print("all_box_acc: ", all_box_acc)
    logger.info(f"the inference on {mode} split of{cfg.DATASET.NAME} result is {all_box_acc}")

    synchronize()
    evaluator.synchronize_between_processes()
    if is_main_process():
        logger.info(f"Complete the inference on {mode} split of {cfg.DATASET.NAME}")

    res = evaluator.summarize()
    return res