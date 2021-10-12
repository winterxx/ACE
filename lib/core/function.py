import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix

import numpy as np
import torch
import torch.distributed as dist
import time
from tqdm import tqdm
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random

def train_model(
    trainLoader, model, epoch, epoch_number, tune_epoch_number, optimizer, combiner, combiner_default, criterion, cfg, logger, rank=0, **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    trainLoader.dataset.update(epoch)
    combiner.update(epoch)
    for c in criterion:
        c.update(epoch)


    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    many_acc = AverageMeter()
    medium_acc = AverageMeter()
    few_acc = AverageMeter()

    for i, (image, label, meta) in enumerate(trainLoader):


        cnt = label.shape[0]
        loss, [now_acc,now_many_acc,now_medium_acc,now_few_acc] = combiner.forward(model, criterion, image, label, meta)

        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        optimizer.step()
        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)
        many_acc.update(now_many_acc, cnt)
        medium_acc.update(now_medium_acc, cnt)
        few_acc.update(now_few_acc, cnt)


        if i % cfg.SHOW_STEP == 0 and rank == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%  Many:{:>5.2f}%  Medium:{:>5.2f}%  Few:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100, many_acc.val * 100, medium_acc.val * 100,
                                                      few_acc.val * 100
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%  Many:{:>5.2f}%  Medium:{:>5.2f}%  Few:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number+tune_epoch_number, all_loss.avg, acc.avg * 100, many_acc.avg * 100, medium_acc.avg * 100, few_acc.avg * 100,
                                           (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)
    return acc.avg, all_loss.avg, many_acc.avg, medium_acc.avg, few_acc.avg



def train_model_group(
    trainLoader, model, epoch, epoch_number, tune_epoch_number, optimizer, combiner, combiner_default, criterion, cfg, logger, rank=0, **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()
    trainLoader.dataset.update(epoch)

    if epoch < epoch_number:
        combiner.update(epoch)
    else:
        combiner_default.update(epoch)
    for c in criterion:
        c.update(epoch)
    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    all_many_head_loss = AverageMeter()
    all_medium_head_loss = AverageMeter()
    all_few_head_loss = AverageMeter()

    acc = AverageMeter()
    many_acc = AverageMeter()
    medium_acc = AverageMeter()
    few_acc = AverageMeter()

    for i, (image, label, meta) in enumerate(trainLoader):
        cnt = label.shape[0]
        time_1 = time.time()
        if epoch<epoch_number:
            loss, [now_acc,now_many_acc,now_medium_acc,now_few_acc] = combiner.forward(model, criterion, image, label, meta)

        else:
            loss, [now_acc, now_many_acc, now_medium_acc, now_few_acc] = combiner_default.forward(model, criterion, image,
                                                                                          label, meta)
        [many_head_loss, medium_head_loss, few_head_loss] = loss
        loss = many_head_loss + medium_head_loss + few_head_loss

        optimizer.zero_grad()
        if cfg.TRAIN.USE_AMP:
            with amp.scale_loss(many_head_loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            many_head_loss.backward(retain_graph=True)

        for name, param in model.named_parameters():
            if not ('medium' in name or 'few' in name):
                param.requires_grad = False
        if cfg.TRAIN.USE_AMP:
            with amp.scale_loss((medium_head_loss+few_head_loss), optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            (medium_head_loss+few_head_loss).backward()

        for name, param in model.named_parameters():
            param.requires_grad = True
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        all_many_head_loss.update(many_head_loss.data.item(), cnt)
        if medium_head_loss != 0:
            all_medium_head_loss.update(medium_head_loss.data.item(), cnt)
        else:
            all_medium_head_loss.update(0, cnt)

        if few_head_loss != 0:
            all_few_head_loss.update(few_head_loss.data.item(), cnt)
        else:
            all_few_head_loss.update(0, cnt)

        acc.update(now_acc, cnt)
        many_acc.update(now_many_acc, cnt)
        medium_acc.update(now_medium_acc, cnt)
        few_acc.update(now_few_acc, cnt)

        if i % cfg.SHOW_STEP == 0 and rank == 0:
            pbar_str1 = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%  Many:{:>5.2f}%  Medium:{:>5.2f}%  Few:{:>5.2f}%   Batch_Time:{:>5.2f}min".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100, many_acc.val * 100, medium_acc.val * 100,
                                                      few_acc.val * 100, (time.time() - time_1) / 60
            )
            logger.info(pbar_str1)

    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%  Many:{:>5.2f}%  Medium:{:>5.2f}%  Few:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number+tune_epoch_number, all_loss.avg, acc.avg * 100, many_acc.avg * 100, medium_acc.avg * 100, few_acc.avg * 100,
                                           (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)
    return acc.avg, all_loss.avg, many_acc.avg, medium_acc.avg, few_acc.avg

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def valid_model(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, rank, distributed, expert_idx, real_expert_idx, **kwargs
):
    model.eval()

    many_idx, medium_idx, few_idx = expert_idx
    real_many_idx, real_medium_idx, real_few_idx = real_expert_idx

    from collections import OrderedDict
    new_dict = OrderedDict()
    for k, v in model.named_parameters():
        if k.startswith("module"):
            new_dict[k] = v

    weight_many = new_dict['module.classifier_many.fc.weight'].detach().cpu().numpy()
    weight_medium = new_dict['module.classifier_medium.fc.weight'].detach().cpu().numpy()
    weight_few = new_dict['module.classifier_few.fc.weight'].detach().cpu().numpy()

    weight_norm_many = LA.norm(weight_many, axis=1)
    weight_norm_medium = LA.norm(weight_medium, axis=1)
    weight_norm_few = LA.norm(weight_few, axis=1)

    m_scale = (np.mean(weight_norm_many) / np.mean(weight_norm_medium[medium_idx[0]:]))
    f_scale = (np.mean(weight_norm_many) / np.mean(weight_norm_few[few_idx[0]:]))

    with torch.no_grad():
        all_loss = AverageMeter()
        acc_avg = AverageMeter()
        many_acc_avg = AverageMeter()
        medium_acc_avg = AverageMeter()
        few_acc_avg = AverageMeter()

        func = torch.nn.Sigmoid() \
            if cfg.LOSS.LOSS_TYPE in ['FocalLoss', 'ClassBalanceFocal'] else \
            torch.nn.Softmax(dim=1)

        for i, (image, label, meta) in enumerate(dataLoader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            final_output = torch.cat(
                (output[0][:, many_idx], (output[0][:, medium_idx] + output[1][:, medium_idx] * m_scale) / 2,
                 (output[0][:, few_idx] + output[1][:, few_idx] * m_scale + output[2][:, few_idx] * f_scale) / 3),1)

            output.append(final_output)
            loss,_ = criterion[0](output, label)

            score_result = func(final_output)

            loss = sum(loss)
            now_result = torch.argmax(score_result, 1)

            acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            many_idx_a = np.where(np.in1d(label.cpu().numpy(), real_many_idx))[0]
            medium_idx_a = np.where(np.in1d(label.cpu().numpy(), real_medium_idx))[0]
            few_idx_a = np.where(np.in1d(label.cpu().numpy(), real_few_idx))[0]

            many_acc, many_cnt = accuracy(now_result.cpu().numpy()[many_idx_a], label.cpu().numpy()[many_idx_a])
            medium_acc,medium_cnt = accuracy(now_result.cpu().numpy()[medium_idx_a], label.cpu().numpy()[medium_idx_a])
            few_acc,few_cnt = accuracy(now_result.cpu().numpy()[few_idx_a], label.cpu().numpy()[few_idx_a])

            if distributed:
                world_size = float(os.environ.get("WORLD_SIZE", 1))
                reduced_loss = reduce_tensor(loss.data, world_size)
                reduced_acc = reduce_tensor(torch.from_numpy(np.array([acc])).cuda(), world_size)
                loss = reduced_loss.cpu().data
                acc = reduced_acc.cpu().data
                reduced_many_acc = reduce_tensor(torch.from_numpy(np.array([many_acc])).cuda(), world_size)
                reduced_medium_acc = reduce_tensor(torch.from_numpy(np.array([medium_acc])).cuda(), world_size)
                reduced_few_acc = reduce_tensor(torch.from_numpy(np.array([few_acc])).cuda(), world_size)

                many_acc = reduced_many_acc.cpu().data
                medium_acc = reduced_medium_acc.cpu().data
                few_acc = reduced_few_acc.cpu().data

            all_loss.update(loss.data.item(), label.shape[0])
            if distributed:
                acc_avg.update(acc.data.item(), cnt * world_size)
                many_acc_avg.update(many_acc.data.item(), many_cnt * world_size)
                medium_acc_avg.update(medium_acc.data.item(), medium_cnt * world_size)
                few_acc_avg.update(few_acc.data.item(), few_cnt * world_size)
            else:
                acc_avg.update(acc, cnt)
                many_acc_avg.update(many_acc, many_cnt)
                medium_acc_avg.update(medium_acc, medium_cnt)
                few_acc_avg.update(few_acc, few_cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%   Many:{:>5.2f}%   Medium:{:>5.2f}%   Few:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc_avg.avg * 100, many_acc_avg.avg * 100, medium_acc_avg.avg * 100,
                                        few_acc_avg.avg * 100
        )
        if rank == 0:
            logger.info(pbar_str)

    return acc_avg.avg, all_loss.avg, many_acc_avg.avg, medium_acc_avg.avg, few_acc_avg.avg