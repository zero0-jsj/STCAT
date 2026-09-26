import argparse
import os
import time
import datetime
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import sys
import json
import numpy as np
import pickle
import random

base_path = os.getcwd()
sys.path.append(base_path)

from config import cfg
from utils.comm import synchronize, get_rank, is_main_process, reduce_loss_dict
from utils.logger import setup_logger
from utils.misc import mkdir, save_config, set_seed, to_device
from utils.checkpoint import VSTGCheckpointer
from datasets import make_data_loader, build_evaluator, build_dataset
from models import build_model, build_postprocessors
from engine import make_optimizer, adjust_learning_rate, update_ema, do_eval
from utils.metric_logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter
from fasterrcnn.get_frcnn_feature import get_frcnn_feature, frcnn_load
from STVMR import STVMR
from util import loss as stvmr_loss, gauss_nce_loss
from stanfordcorenlp import StanfordCoreNLP
# from util import nce_loss as stvmr_loss
from util import get_top_20_region_feature


def train(cfg, local_rank, distributed, logger):
    torch.cuda.empty_cache()
    device = torch.device(cfg.MODEL.DEVICE)

    arguments = {}
    arguments["iteration"] = 0

    # Prepare the dataset cache
    if local_rank == 0:
        split = ['train', 'test']
        if cfg.DATASET.NAME == "VidSTG":
            split += ['val']
        for mode in split:
            _ = build_dataset(cfg, split=mode, transforms=None)

    synchronize()

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loader = make_data_loader(
        cfg,
        mode='val' if cfg.DATASET.NAME == "VidSTG" else "test",
        is_distributed=distributed,
    )

    if cfg.TENSORBOARD_DIR and is_main_process():
        writer = SummaryWriter(cfg.TENSORBOARD_DIR)
    else:
        writer = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    logger.info("Start training")

    metric_logger = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    fasterRCNN = frcnn_load()
    nlp = StanfordCoreNLP(cfg.BERT.CORENLP_PAYH)

    stvmr = STVMR(cfg.STVMR.SENTENCE_SHAPE, cfg.STVMR.REGION_SHAPE, cfg.STVMR.FEATURE_SHAPE, cfg.BERT.BERT_PATH).to(device)
    optimizer = torch.optim.Adam(stvmr.parameters(), lr=cfg.STVMR.LR, weight_decay=cfg.STVMR.WEIGHT_DECAY)

    model_ema = deepcopy(stvmr) if cfg.MODEL.EMA else None
    model_without_ddp = stvmr

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = local_rank == 0
    checkpointer = VSTGCheckpointer(
        cfg, model_without_ddp, model_ema, optimizer, output_dir, save_to_disk, logger, is_train=True
    )

    # extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, with_optim=False)
    # arguments.update(extra_checkpoint_data)


    for iteration, batch_dict in enumerate(train_data_loader, start_iter):


        iteration = iteration + 1
        data_time = time.time() - end
        arguments["iteration"] = iteration

        videos = batch_dict['videos'].to(device)
        texts = batch_dict['texts']
        targets = to_device(batch_dict["targets"], device)
        video = videos.tensors
        # print(video.shape)
        # print(texts)
        # print(targets)

        rois = []
        batch_anno_frame = []
        for batch in range(len(targets)):
            actions = targets[batch]['actioness']
            boxes = targets[batch]['boxs'].bbox
            no_zero_inedx = torch.nonzero(actions)
            # print(actions)
            # print(no_zero_inedx[0][0],no_zero_inedx[-1][0])
            anno_f = random.randint(no_zero_inedx[0][0], no_zero_inedx[-1][0])
            # print(anno_f)
            j = anno_f - no_zero_inedx[0][0]
            # print(j)
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
            roi[0] = anno_f + batch * cfg.INPUT.TRAIN_SAMPLE_NUM
            batch_anno_frame.append(anno_f)
            # print(roi)
            rois.append(roi)
        rois = torch.stack(rois).to(device)
        batch_anno_frame = torch.tensor(batch_anno_frame)
        # print(rois.shape)
        # print(batch_anno_frame)

        frccn_feature, anno_index, rois = get_frcnn_feature(video, rois, fasterRCNN, True)
        # print(frccn_feature.shape)

        region_feature = get_top_20_region_feature(frccn_feature, anno_index, batch_anno_frame)
        # print(region_feature)
        region_feature = region_feature.to(device)

        sen_feature, noun_word_mask = stvmr.get_bert_feature(nlp, texts)
        sen_feature = sen_feature.to(device)
        noun_word_mask = torch.tensor(noun_word_mask)
        noun_word_mask = noun_word_mask.to(device)
        # print(sen_feature)
        # print(noun_word_mask.shape)

        if not cfg.STVMR.IS_GAUSS_LOSS:
            spatio_similarity_socre, tem_similarity_score = stvmr(sen_feature, region_feature, batch_anno_frame + 1)
        else:
            spatio_similarity_socre, tem_similarity_score, video_length, sigma, mu, clip_information = stvmr(sen_feature, region_feature, batch_anno_frame + 1)
        # print("score: ")
        # print(spatio_similarity_socre)
        # print(tem_similarity_score)

        if not cfg.STVMR.IS_GAUSS_LOSS:
            loss = stvmr_loss(spatio_similarity_socre, tem_similarity_score, noun_word_mask, cfg.STVMR.DELTA_TEMPORAL,
                              cfg.STVMR.DELTA_SPATIO, cfg.STVMR.LAMDA)
        else:
            loss = gauss_nce_loss(spatio_similarity_socre, tem_similarity_score, noun_word_mask, video_length, sigma, mu, clip_information)
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adjust_learning_rate(cfg, optimizer, iteration, max_iter)
        if model_ema is not None:
            update_ema(stvmr, model_ema, cfg.MODEL.EMA_DECAY)

        batch_time = time.time() - end
        end = time.time()
        metric_logger.update(time=batch_time, data=data_time)

        eta_seconds = metric_logger.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 50 == 0 or iteration == max_iter:
            logger.info(
                metric_logger.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter} / {max_iter}",
                        "{meters}",
                        "loss: {loss}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    max_iter=max_iter,
                    loss=loss,
                    meters=str(metric_logger),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % (max_iter // 10) == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
        # break
        # run_eval(cfg, stvmr, model_ema, logger, val_data_loader, device)
        # break
        if cfg.SOLVER.TO_VAL and iteration % (max_iter // 10) == 0:
            run_eval(cfg, stvmr, model_ema, logger, val_data_loader, device)
            # run_test(cfg, model, model_ema, logger, distributed)
        # if cfg.SOLVER.TO_VAL and iteration == 1:
        #     run_eval(cfg, stvmr, model_ema, logger, val_data_loader, device)

        # break

        # first_f = 0
        # end_f = len(actions)
        # flag = False
        # for i in range(len(actions)):
        #     if actions[i] == 0 and flag:
        #         end_f = i
        #         break
        #     if actions[i] == 1 and not flag:
        #         first_f = i
        #         flag = True
        #
        # anno_f = random.randint(first_f, end_f-1)
        # j = anno_f - first_f
        # boxes = targets[0]['boxs'].bbox

        # text = texts[0]
        # # stvmr.do_eval(video, text, device)
        # anno_box = boxes[j]
        # image_size = video.shape[-1]
        # roi = torch.zeros([5])
        # roi[1] = (anno_box[0] - anno_box[2]*0.5)
        # roi[2] = (anno_box[1] - anno_box[3]*0.5)
        # roi[3] = (anno_box[0] + anno_box[2]*0.5)
        # roi[4] = (anno_box[1] + anno_box[3]*0.5)
        # roi = roi*image_size
        # roi[0] = anno_f
        #
        # batch_text.append(text)
        # batch_anno_frame.append(anno_f)

        # frccn_feature, anno_index, rois = get_frcnn_feature(video, roi, fasterRCNN, True)
        #
        # region_feature = get_top_20_region_feature(frccn_feature, anno_index, anno_f)
        #
        # # region_feature = region_feature.unsqueeze(0)
        # # print(region_feature.shape)
        #
        # batch_region_feature.append(region_feature)
        #
        # # break
        #
        # if iteration % cfg.STVMR.BATCH_SIZE == 0:
        #     # print(iteration)
        #     # print(batch_text)
        #     # print(batch_anno_frame)
        #     batch_anno_frame = torch.tensor(batch_anno_frame)
        #     batch_anno_frame = batch_anno_frame.to(device)
        #     batch_region_feature = torch.stack(batch_region_feature)
        #     batch_region_feature = batch_region_feature.to(device)
        #     # print(batch_region_feature.shape)
        #
        #     sen_feature, noun_word_mask = stvmr.get_bert_feature(nlp, batch_text)
        #     sen_feature = sen_feature.to(device)
        #     noun_word_mask = torch.tensor(noun_word_mask)
        #     noun_word_mask = noun_word_mask.to(device)
        #     # print(sen_feature.shape)
        #     # print(noun_word_mask.shape)
        #
        #     spatio_similarity_socre, tem_similarity_score = stvmr(sen_feature, batch_region_feature, batch_anno_frame+1)
        #
        #     # print(spatio_similarity_socre)
        #     # print(tem_similarity_score)
        #
        #     loss = stvmr_loss(spatio_similarity_socre, tem_similarity_score, noun_word_mask, cfg.STVMR.DELTA_TEMPORAL, cfg.STVMR.DELTA_SPATIO, cfg.STVMR.LAMDA)
        #     # print(loss)
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        #     # adjust_learning_rate(cfg, optimizer, iteration, max_iter)
        #     if model_ema is not None:
        #         update_ema(stvmr, model_ema, cfg.MODEL.EMA_DECAY)
        #
        #     batch_time = time.time() - end
        #     end = time.time()
        #     metric_logger.update(time=batch_time, data=data_time)
        #
        #     eta_seconds = metric_logger.time.global_avg * (max_iter - iteration)
        #     eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        #
        #     if iteration % (50*cfg.STVMR.BATCH_SIZE) == 0 or iteration == max_iter:
        #         logger.info(
        #             metric_logger.delimiter.join(
        #                 [
        #                     "eta: {eta}",
        #                     "iter: {iter} / {max_iter}",
        #                     "{meters}",
        #                     "loss: {loss}",
        #                     "lr: {lr:.6f}",
        #                     "max mem: {memory:.0f}",
        #                 ]
        #             ).format(
        #                 eta=eta_string,
        #                 iter=iteration,
        #                 max_iter=max_iter,
        #                 loss=loss,
        #                 meters=str(metric_logger),
        #                 lr=optimizer.param_groups[0]["lr"],
        #                 memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
        #             )
        #         )
        #
        #     batch_region_feature = []
        #     batch_text = []
        #     batch_anno_frame = []
        #     # break
        #
        # if iteration % (max_iter // 10) == 0:
        #     checkpointer.save("model_{:06d}".format(iteration), **arguments)
        #
        # if iteration == max_iter:
        #     checkpointer.save("model_final", **arguments)
        # # break
        # # run_eval(cfg, stvmr, model_ema, logger, val_data_loader, device)
        # # break
        # if cfg.SOLVER.TO_VAL and iteration % (max_iter // 10) == 0:
        #     run_eval(cfg, stvmr, model_ema, logger, val_data_loader, device)
        #     # run_test(cfg, model, model_ema, logger, distributed)
        # if cfg.SOLVER.TO_VAL and iteration == 1:
        #     run_eval(cfg, stvmr, model_ema, logger, val_data_loader, device)


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    if writer is not None:
        writer.close()

    return stvmr, model_ema

def run_eval(cfg, model, model_ema, logger, val_data_loader, device):
    logger.info("Start validating")
    test_model = model_ema if model_ema is not None else model
    evaluator = build_evaluator(cfg, logger, mode='val' \
        if cfg.DATASET.NAME == "VidSTG" else "test", )  # mode = ['val','test']
    postprocessor = build_postprocessors()
    torch.cuda.empty_cache()
    do_eval(
        cfg,
        mode='val',
        logger=logger,
        model=test_model,
        postprocessor=postprocessor,
        data_loader=val_data_loader,
        evaluator=evaluator,
        device=device
    )
    synchronize()


def run_test(cfg, model, model_ema, logger, distributed):
    logger.info("Start Testing")
    test_model = model_ema if model_ema is not None else model
    torch.cuda.empty_cache()

    evaluator = build_evaluator(cfg, logger, mode='test')  # mode = ['val','test']
    postprocessor = build_postprocessors()
    val_data_loader = make_data_loader(cfg, mode='test', is_distributed=distributed)
    do_eval(
        cfg,
        mode='test',
        logger=logger,
        model=test_model,
        postprocessor=postprocessor,
        data_loader=val_data_loader,
        evaluator=evaluator,
        device=torch.device(cfg.MODEL.DEVICE)
    )
    synchronize()


def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Grounding Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--use-seed",
        dest="use_seed",
        help="If use the random seed",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.use_seed:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed + get_rank())

    synchronize()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("Video Grounding", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))

    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model, model_ema = train(cfg, args.local_rank, args.distributed, logger)

    # if not args.skip_test:
        # run_test(cfg, model, model_ema, logger, args.distributed)


if __name__ == "__main__":
    main()
