import _init_paths

from loss import *
from dataset import *
from config import cfg, update_config
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_category_list,
    get_category_list_hierarchical,
)
from utils.cam_based_sampling import cam_based_sampling
from core.function import train_model,train_model_group,valid_model
from core.combiner import Combiner

import torch
import os
import shutil
from torch.utils.data import DataLoader
import argparse
import warnings
import click
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import ast
from datetime import datetime

import apex
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import random
import numpy as np

from flopth import flopth

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="codes for BagofTricks-LT")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar10_im100.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type= ast.literal_eval,
        dest = 'auto_resume',
        required=False,
        default= True,
    )

    parser.add_argument(
        "--local_rank",
        help='local_rank for distributed training',
        type=int,
        default=0,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    rank = local_rank
    update_config(cfg, args)
    logger, log_file = create_logger(cfg, local_rank)
    warnings.filterwarnings("ignore")
    auto_resume = args.auto_resume

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cudnn.deterministic = True
    cudnn.benchmark = False

    # close loop
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models",
                             str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes",
                             str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard",
                             str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
        if cfg.TRAIN.TENSORBOARD.ENABLE
        else None
    )
    tsne_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tsne",
                 str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
    if local_rank == 0:

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )
            if not click.confirm(
                "\033[1;31;40mContinue and override the former directory?\033[0m",
                default=False,
            ):
                exit(0)
            shutil.rmtree(code_dir)
            if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)
        print("=> output model will be saved in {}".format(model_dir))
        #import pdb;pdb.set_trace()
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        try:
            #shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)
            pass
        except:
            pass

    if cfg.TRAIN.DISTRIBUTED:
        if local_rank == 0:
            print('Init the process group for distributed training')
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        if local_rank == 0:
            print('Init complete')

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)

    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()
    num_super_classes = train_set.get_num_super_classes()
    device = torch.device("cuda")


    num_class_list, cat_list = get_category_list(annotations, num_classes, cfg)
    sup_num_list = num_class_list
    num_super_classes = num_classes

    residual = []
    for i in range(len(num_class_list)):
        residual.append(sum(num_class_list[i:])/float(sum(num_class_list)))

    experts_split =  cfg.TRAIN.OPTIMIZER.EXPERT
    expert_idx = []

    for i in experts_split:
        for j in range(len(residual)-1):
            if residual[j]>i and residual[j+1]<=i:
                expert_idx.append(j)
                break

    real_many_idx = torch.tensor(np.argwhere(np.asarray(num_class_list) > 100).squeeze())
    real_medium_idx = torch.tensor(np.argwhere((np.asarray(num_class_list) > 20) & (np.asarray(num_class_list) <= 100)).squeeze())
    real_few_idx = torch.tensor(np.argwhere(np.asarray(num_class_list) <= 20).squeeze())

    many_idx = torch.tensor(np.asarray(range(expert_idx[0])))
    medium_idx = torch.tensor(np.asarray(range(expert_idx[0], expert_idx[1])))
    few_idx = torch.tensor(np.asarray(range(expert_idx[1], len(num_class_list))))

    para_dict = {
        "num_classes": num_classes,
        "num_super_classes": num_super_classes,
        "num_class_list": num_class_list,
        "num_superclass_list":sup_num_list,
        "cfg": cfg,
        "device": device,
    }
    print(cfg.LOSS.METRIC_LOSS_TYPE )
    criterion = [eval(cfg.LOSS.LOSS_TYPE)([many_idx, medium_idx, few_idx], para_dict=para_dict)]
    if cfg.LOSS.METRIC_LOSS_TYPE != '':
        criterion.append(eval(cfg.LOSS.METRIC_LOSS_TYPE)([many_idx, medium_idx, few_idx], para_dict=para_dict))

    epoch_number = cfg.TRAIN.MAX_EPOCH
    unmix_epoch_number = cfg.TRAIN.UNMIX_EPOCH

    # ----- BEGIN MODEL BUILDER -----
    model = get_model(cfg, num_class_list, num_classes, num_super_classes, device, logger)
    combiner = Combiner(cfg, cfg.TRAIN.COMBINER.TYPE, device, num_class_list, [many_idx, medium_idx, few_idx], [real_many_idx, real_medium_idx, real_few_idx])
    combiner_default = Combiner(cfg, 'default', device, num_class_list, [many_idx, medium_idx, few_idx],
                        [real_many_idx, real_medium_idx, real_few_idx])
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    if cfg.DATASET.GENERATE_CAM_BASED_DATASET:
        cam_based_sampling(train_set, model, cfg)
        exit(0)

    opt_level = 'O1'
    if cfg.TRAIN.DISTRIBUTED:
        model = model.cuda()
        model = apex.parallel.convert_syncbn_model(model)
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        model = DDP(model, delay_allreduce=True)
    else:
        if cfg.TRAIN.USE_AMP:
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        model = torch.nn.DataParallel(model, output_device=1)

    # = flopth(model, in_size=[[3, 224, 224]])
    #print(sum_flops)

    # ----- END MODEL BUILDER -----

    if cfg.TRAIN.DISTRIBUTED:
        train_sampler = torch.utils.data.DistributedSampler(train_set)
        val_sampler = torch.utils.data.DistributedSampler(valid_set)
        trainLoader = DataLoader(
            train_set,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,
            sampler=train_sampler,
        )
        validLoader = DataLoader(
            valid_set,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            pin_memory=False,
            sampler=val_sampler,
        )

    else:
        trainLoader = DataLoader(
            train_set,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=True
        )

        #import time
        #since = time.time()
        #imgs = next(iter(trainLoader))
        #cost = time.time() - since
        #print("loader", cost)
        #import pdb;pdb.set_trace()
        validLoader = DataLoader(
            valid_set,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )


    if tensorboard_dir is not None and local_rank == 0:
        dummy_input = torch.rand((1, 3) + cfg.INPUT_SIZE).to(device)
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        writer = None

    best_result, best_epoch, start_epoch = 0, 0, 1
    # ----- BEGIN RESUME ---------
    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume == False:
        auto_resume = False
    else:
        all_models.remove("best_model.pth")
        resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir, "epoch_{}.pth".format(resume_epoch))

    if cfg.RESUME_MODEL != "" or auto_resume:
        if cfg.RESUME_MODEL == "":
            resume_model = resume_model_path
        else:
            resume_model = cfg.RESUME_MODEL if '/' in cfg.RESUME_MODEL else os.path.join(model_dir, cfg.RESUME_MODEL)
        logger.info("Loading checkpoint from {}...".format(resume_model))
        checkpoint = torch.load(
            resume_model, map_location="cpu" if cfg.TRAIN.DISTRIBUTED else "cuda"
        )
        model.module.load_model(resume_model)
        if cfg.RESUME_MODE != "state_dict":
            #optimizer.load_state_dict(checkpoint['optimizer'])
            #scheduler.load_state_dict(checkpoint['scheduler'])
            #amp.load_state_dict(checkpoint['amp'])
            #start_epoch = checkpoint['epoch'] + 1
            start_epoch = 0
            #best_result = checkpoint['best_result']
            #best_epoch = checkpoint['best_epoch']
    # ----- END RESUME ---------

    if rank == 0:
        logger.info(
            "-------------------Train start :{}  {}  {}-------------------".format(
                cfg.BACKBONE.TYPE, cfg.MODULE.TYPE, cfg.TRAIN.COMBINER.TYPE
            )
        )

    for epoch in range(start_epoch, epoch_number + unmix_epoch_number + 1):
        if cfg.TRAIN.DISTRIBUTED:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        scheduler.step()
        print(epoch, scheduler.get_last_lr())

        if 'group' in cfg.BACKBONE.TYPE:

            train_acc, train_loss, many_acc, medium_acc, few_acc = train_model_group(
                trainLoader,
                model,
                epoch,
                epoch_number,
                unmix_epoch_number,
                optimizer,
                combiner,
                combiner_default,
                criterion,
                cfg,
                logger,
                writer=writer,
                rank=local_rank,
            )

        else:
            train_acc, train_loss, many_acc,medium_acc, few_acc = train_model(
                trainLoader,
                model,
                epoch,
                epoch_number,
                unmix_epoch_number,
                optimizer,
                combiner,
                combiner_default,
                criterion,
                cfg,
                logger,
                writer=writer,
                rank=local_rank,
            )
        model_save_path = os.path.join(
            model_dir,
            "epoch_{}.pth".format(epoch),
        )
        if epoch % cfg.SAVE_STEP == 0 and local_rank == 0:
            if isinstance(optimizer, list):
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer[0].state_dict()
                }, model_save_path)
            else:
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_save_path)

        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
        acc_dict["train_many_acc"] = many_acc
        acc_dict["train_medium_acc"]=  medium_acc
        acc_dict["train_few_acc"]=few_acc

        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            valid_acc, valid_loss,valid_many_acc,valid_medium_acc, valid_few_acc  = valid_model(
                validLoader, epoch, model, cfg, criterion, logger, device,
                rank=rank, distributed=cfg.TRAIN.DISTRIBUTED, expert_idx= [many_idx, medium_idx, few_idx],
                real_expert_idx=[real_many_idx, real_medium_idx, real_few_idx], writer=writer)
        loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc
        acc_dict["valid_many_acc"] = valid_many_acc
        acc_dict["valid_medium_acc"] = valid_medium_acc
        acc_dict["valid_few_acc"] = valid_few_acc

        if valid_acc > best_result and local_rank == 0:
            best_result, best_epoch = valid_acc, epoch

            if isinstance(optimizer, list):
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer[0].state_dict()
                }, os.path.join(model_dir, "best_model.pth")
                )
            else:
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model.pth")
                )
        if rank == 0:
            logger.info(
                "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                    best_epoch, best_result * 100
                )
            )

        if cfg.TRAIN.TENSORBOARD.ENABLE and local_rank == 0:
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
if cfg.TRAIN.TENSORBOARD.ENABLE and local_rank == 0:
    writer.close()
if rank == 0:
    logger.info(
        "-------------------Train Finished :{}-------------------".format(cfg.NAME)
    )
