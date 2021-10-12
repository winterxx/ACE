import _init_paths
from net import Network,Network_Group,Network_Group_Ntire
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile
import argparse
from core.evaluate import FusionMatrix
from flopth import flopth
from utils.utils import (
    get_category_list,
)
import matplotlib.pyplot as plt
from numpy import linalg as LA
def parse_args():
    parser = argparse.ArgumentParser(description="tricks evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="configs/cifar10_im100.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def valid_model(dataLoader, model, cfg, device, num_classes, num_class_list, expert_idx, real_expert_idx, fea_file=None, pred_file=None, label_file=None):
    result_list = []

    [many_idx, medium_idx, few_idx] = expert_idx
    [real_many_idx, real_medium_idx, real_few_idx] = real_expert_idx

    pbar = tqdm(total=len(dataLoader))
    model.eval()
    top1_count, top2_count, top3_count, index, fusion_matrix = (
        [],
        [],
        [],
        0,
        FusionMatrix(num_classes),
    )

    func = torch.nn.Softmax(dim=1)

    classpred = []
    classlabel = []
    #feature_all = []
    result_all = []

    many = []
    medium = []
    few = []
    many_img = 0
    medium_img = 0
    few_img = 0


    from collections import OrderedDict
    new_dict = OrderedDict()
    for k, v in model.named_parameters():
        if k.startswith("module"):
            new_dict[k] = v
    weight_many = new_dict['module.classifier_many.fc.weight'].cpu().detach().numpy()
    weight_medium = new_dict['module.classifier_medium.fc.weight'].cpu().detach().numpy()
    weight_few = new_dict['module.classifier_few.fc.weight'].cpu().detach().numpy()
    scale_many = new_dict['module.classifier_many.scales'].cpu().detach().numpy()
    scale_medium = new_dict['module.classifier_medium.scales'].cpu().detach().numpy()
    scale_few = new_dict['module.classifier_few.scales'].cpu().detach().numpy()

    weight_norm_many = LA.norm(weight_many, axis=1)
    weight_norm_medium = LA.norm(weight_medium, axis=1)
    weight_norm_few = LA.norm(weight_few, axis=1)

    #import pdb;pdb.set_trace()
    #m_scale = (np.mean(weight_norm_many) / np.mean(weight_norm_medium[medium_idx[0]:])) * (np.mean(scale_many) / np.mean(scale_medium[medium_idx[0]:]))
    #f_scale = (np.mean(weight_norm_many) / np.mean(weight_norm_few[few_idx[0]:])) * (np.mean(scale_many) / np.mean(scale_few[few_idx[0]:]))
    m_scale = (np.mean(weight_norm_many) / np.mean(weight_norm_medium[medium_idx[0]:]))
    f_scale = (np.mean(weight_norm_many) / np.mean(weight_norm_few[few_idx[0]:]))
    #print(m_scale, f_scale)
    #m_scale = f_scale  =1

    with torch.no_grad():
        for i, (image, image_labels, meta) in enumerate(dataLoader):
            image = image.to(device)
            output = model(image, train=False)
            output = torch.cat((output[0][:, many_idx], (output[0][:, medium_idx] + output[1][:, medium_idx] * m_scale) / 2,(output[0][:, few_idx] + output[1][:, few_idx] * m_scale + output[2][:,few_idx] * f_scale) / 3), 1)
            #output = output[0]
            #feature = model(image, train=False, feature_flag=True)[0]
            result = func(output)
            _, top_k = result.topk(5, 1, True, True)
            score_result = result.cpu().numpy()

            fusion_matrix.update(score_result.argmax(axis=1), image_labels.numpy())
            topk_result = top_k.cpu().tolist()
            classpred += list(np.asarray(topk_result)[:, 0])

            classlabel += list(image_labels.numpy())
            if i == 0:
                result_all = result.cpu().numpy()
            else:
                result_all = np.vstack((result_all, result.cpu().numpy()))
            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                result_list.append(
                    {
                        "image_id": image_id,
                        "image_label": int(image_labels[i]),
                        "top_3": topk_result[i],
                    }
                )
                top1_count += [topk_result[i][0] == image_labels[i]]

                top2_count += [image_labels[i] in topk_result[i][0:2]]
                top3_count += [image_labels[i] in topk_result[i][0:3]]
                index += 1
            now_acc = np.sum(top1_count) / index
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)
    top1_acc = float(np.sum(top1_count) / len(top1_count))
    top2_acc = float(np.sum(top2_count) / len(top1_count))
    top3_acc = float(np.sum(top3_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100
        )
    )
    pbar.close()
    #fig = fusion_matrix.plot_confusion_matrix()
    #plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.NAME,'confusion_matrix.png'))
    #if fea_file is not None:
    #    np.save(fea_file, feature_all)
    #    np.save(pred_file, result_all)
    #    np.save(label_file, classlabel)

    classlabel = np.asarray(classlabel)
    classpred = np.asarray(classpred)
    many_pred = 0
    many_label = 0
    medium_pred = 0
    medium_label = 0
    few_pred = 0
    few_label = 0

    for clsidx in list(range(num_classes)):
        acc = np.logical_and(classpred == clsidx,classpred == classlabel).sum()/np.count_nonzero(classlabel == clsidx)
       # print('{0: <5}'.format(clsidx), ":{:>5.2f}%".format(acc*100))
        if clsidx in real_few_idx:
            few_pred += np.logical_and(classpred == clsidx,classpred == classlabel).sum()
            few_label += np.count_nonzero(classlabel == clsidx)
        elif clsidx in real_medium_idx:
            medium_pred += np.logical_and(classpred == clsidx, classpred == classlabel).sum()
            medium_label += np.count_nonzero(classlabel == clsidx)
        else:
            many_pred += np.logical_and(classpred == clsidx, classpred == classlabel).sum()
            many_label += np.count_nonzero(classlabel == clsidx)

    #for clsidx in real_many_idx:
    #    acc = np.logical_and(classpred == clsidx, classpred == classlabel).sum() / np.count_nonzero(
    #        classlabel == clsidx)
        #print('{0: <5} (many)'.format(clsidx), ":{:>5.2f}%".format(acc * 100))
    #for clsidx in real_medium_idx:
    #    acc = np.logical_and(classpred == clsidx, classpred == classlabel).sum() / np.count_nonzero(
    #        classlabel == clsidx)
        #print('{0: <5} (medium)'.format(clsidx), ":{:>5.2f}%".format(acc * 100))
    #for clsidx in real_few_idx:
    #    acc = np.logical_and(classpred == clsidx, classpred == classlabel).sum() / np.count_nonzero(
    #        classlabel == clsidx)
        #print('{0: <5} (few)'.format(clsidx), ":{:>5.2f}%".format(acc * 100))


    print('{0: <20}'.format('Many'), ":{:>5.2f}%".format(many_pred/many_label*100))
    print('{0: <20}'.format('Medium'), ":{:>5.2f}%".format(medium_pred / medium_label*100))
    print('{0: <20}'.format('Few'), ":{:>5.2f}%".format(few_pred / few_label * 100))




if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()

    device = torch.device("cuda")


    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )

    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()
    num_class_list, cat_list = get_category_list(annotations, num_classes, cfg)
    residual = []
    for i in range(len(num_class_list)):
        residual.append(sum(num_class_list[i:]) / float(sum(num_class_list)))

    experts_split = cfg.TRAIN.OPTIMIZER.EXPERT
    expert_idx = []

    for i in experts_split:
        for j in range(len(residual) - 1):
            if residual[j] > i and residual[j + 1] <= i:
                expert_idx.append(j)
                break

    real_many_idx = torch.tensor(np.argwhere(np.asarray(num_class_list) > 100).squeeze())
    real_medium_idx = torch.tensor(
        np.argwhere((np.asarray(num_class_list) > 20) & (np.asarray(num_class_list) <= 100)).squeeze())
    real_few_idx = torch.tensor(np.argwhere(np.asarray(num_class_list) <= 20).squeeze())

    many_idx = torch.tensor(np.asarray(range(expert_idx[0])))
    medium_idx = torch.tensor(np.asarray(range(expert_idx[0], expert_idx[1])))
    few_idx = torch.tensor(np.asarray(range(expert_idx[1], len(num_class_list))))

    if  'group' in cfg.BACKBONE.TYPE:
        model = Network_Group(cfg, mode="test", num_classes=num_classes)
    else:
        model = Network(cfg, groups=[many_idx,medium_idx,few_idx], mode="test", num_classes=num_classes)

    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE

    if cfg.TEST.FEA_FILE != "":
        fea_file = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, cfg.TEST.FEA_FILE)
        pred_file = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, cfg.TEST.PRED_FILE)
        label_file = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, cfg.TEST.LABEL_FILE)

    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    model.load_model(model_path)

    model = torch.nn.DataParallel(model).cuda()

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )
    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()
    num_class_list, cat_list = get_category_list(annotations, num_classes, cfg)
    valid_model(testLoader, model, cfg, device, num_classes, num_class_list, [many_idx, medium_idx, few_idx],
               [real_many_idx, real_medium_idx, real_few_idx], fea_file, pred_file, label_file)
