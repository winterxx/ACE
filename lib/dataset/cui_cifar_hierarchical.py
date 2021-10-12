from dataset.baseset import BaseSet
import numpy as np
import torch
import json, os, random, time
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
from data_transform.transform_wrapper import TRANSFORMS
from utils.utils import get_category_list_hierarchical
import math
from PIL import Image
import itertools
class CIFAR_HIERARCHICAL(Dataset):
    def __init__(self, mode = 'train', cfg = None, transform = None):
        self.mode = mode
        self.transform = transform
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE
        self.size = self.input_size

        print("Use {} Mode to train network".format(self.color_space))

        if self.mode == "train":
            print("Loading train data ...", end=" ")
            self.json_path = cfg.DATASET.TRAIN_JSON
        elif "valid" in self.mode:
            print("Loading valid data ...", end=" ")
            self.json_path = cfg.DATASET.VALID_JSON
        else:
            raise NotImplementedError
        self.update_transform()

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)

        self.num_classes = self.all_info["num_classes"]
        self.num_super_classes = self.all_info["num_super_classes"]
        self.num_classes_per_super_classes = self.all_info['num_classes_per_super_classes']


        if not self.cfg.DATASET.USE_CAM_BASED_DATASET or self.mode != 'train':
            self.data = self.all_info['annotations']
        else:
            assert os.path.isfile(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH), \
                'the CAM-based generated json file does not exist!'
            self.data = json.load(open(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH))
        print("Contain {} images of {} super_classes ({} sub classes)".format(len(self.data), self.num_super_classes, self.num_classes))

        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and mode == "train":
            self.class_weight, self.sum_weight = self.get_weight(self.data,  self.num_classes_per_super_classes)

            print('-' * 20 + ' dataset' + '-' * 20)
            print('class_weight is (super cls 0): ')
            print(self.class_weight[0])
            print('class_weight is (super cls 1): ')
            print(self.class_weight[1])

            sup_num_list, sup_cat_list, num_list, cat_list = \
                get_category_list_hierarchical(self.get_annotations(), self.num_classes_per_super_classes, self.cfg)
            self.num_list = num_list
            self.hierarchical_supbalance_p = [] #super class balanced
            self.hierarchical_supimbalance_p = []  # super class imbalanced

            self.instance_p = []
            self.class_p = []
            self.super_class_p = []
            self.square_p = []
            self.sup_class_balance_p = [1 / self.num_super_classes for _ in sup_num_list]    #super_class_balance
           # self.sup_class_imbalance_p = [sum(x) / sum(sup_num_list) for x in sup_num_list]  # super_class_imbalance

            #import pdb;pdb.set_trace()
            self.class_p = np.array([1 / sum(self.num_classes_per_super_classes) for _ in range(sum(self.num_classes_per_super_classes))])
            all_sub_list = [0 for j in range(sum(self.num_classes_per_super_classes))]
            for i in range(len(num_list)):
                sup_num_list = num_list[i]
                for k,v in sup_num_list.items():
                    all_sub_list[k] = v

            self.instance_p = np.array([num / sum(all_sub_list) for num in all_sub_list])
            all_sub_list = [math.sqrt(num) for num in all_sub_list]
            self.square_p = np.array([num / sum(all_sub_list) for num in all_sub_list])
            self.instance_p = np.asarray(self.instance_p)
            self.sup_class_dict, self.class_dict = self._get_class_dict()
            #import pdb;pdb.set_trace()

    def update(self, epoch):
        self.epoch = epoch
        if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "progressive" and epoch <= self.cfg.TRAIN.MAX_EPOCH:
            self.progress_p = epoch / self.cfg.TRAIN.MAX_EPOCH * self.class_p + (
                        1 - epoch / self.cfg.TRAIN.MAX_EPOCH) * self.instance_p
        elif self.cfg.TRAIN.TUNE_SAMPLER.WEIGHTED_SAMPLER.TYPE == "progressive" and epoch > self.cfg.TRAIN.MAX_EPOCH:
            self.progress_p = (epoch-self.cfg.TRAIN.MAX_EPOCH) / self.cfg.TRAIN.TUNE_EPOCH * self.class_p + (
                    1 - (epoch-self.cfg.TRAIN.MAX_EPOCH) / self.cfg.TRAIN.TUNE_EPOCH ) * self.instance_p

    def __getitem__(self, index):
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.mode == 'train' and self.epoch <= self.cfg.TRAIN.MAX_EPOCH:
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", 'square', 'progressive',
                                                                    'hierarchical_sup_balance','hierarchical_sup_balance_sub_balance']
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "hierarchical_sup_balance":
                sample_sup_class =  np.random.choice(np.arange(self.num_super_classes), p=self.sup_class_balance_p)
                #print(self.num_list[sample_sup_class])
                p_class = [v/sum(list(self.num_list[sample_sup_class].values())) for k,v in self.num_list[sample_sup_class].items()]
                sample_class = np.random.choice(list(self.num_list[sample_sup_class].keys()), p=p_class)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "hierarchical_sup_balance_sub_balance":
                sample_sup_class = np.random.choice(np.arange(self.num_super_classes), p=self.sup_class_balance_p)
                p_class = [1 / len(list(self.num_list[sample_sup_class].keys())) for k in
                           list(self.num_list[sample_sup_class].keys())]
                sample_class = np.random.choice(list(self.num_list[sample_sup_class].keys()), p=p_class)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)

            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        elif self.cfg.TRAIN.TUNE_SAMPLER.TYPE == "weighted sampler" and self.mode == 'train' and self.epoch > self.cfg.TRAIN.MAX_EPOCH:
            if self.cfg.TRAIN.TUNE_SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.cfg.TRAIN.TUNE_SAMPLER.WEIGHTED_SAMPLER.TYPE == "square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            elif self.cfg.TRAIN.TUNE_SAMPLER.WEIGHTED_SAMPLER.TYPE == "hierarchical_sup_balance":
                sample_sup_class =  np.random.choice(np.arange(self.num_super_classes), p=self.sup_class_balance_p)
                p_class = [v/sum(list(self.num_list[sample_sup_class].values())) for k,v in self.num_list[sample_sup_class].items()]
                sample_class = np.random.choice(list(self.num_list[sample_sup_class].keys()), p=p_class)
            elif self.cfg.TRAIN.TUNE_SAMPLER.WEIGHTED_SAMPLER.TYPE == "hierarchical_sup_balance_sub_balance":
                sample_sup_class = np.random.choice(np.arange(self.num_super_classes), p=self.sup_class_balance_p)
                p_class = [1 / len(list(self.num_list[sample_sup_class].keys())) for k in
                           list(self.num_list[sample_sup_class].keys())]
                sample_class = np.random.choice(list(self.num_list[sample_sup_class].keys()), p=p_class)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)

            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)
        meta = dict()
        image_label = now_info['category_id']  # 0-index
        image_label_super = now_info['super_category_id']  # 0-index
        return image, image_label_super, image_label, meta

    def get_weight(self, annotations,  num_classes_per_super_classes):
        num_superclass = len(num_classes_per_super_classes)
        num_list = [{} for i in range(num_superclass)]

        cat_list = [[] for j in range(num_superclass)]
        for anno in annotations:
            super_category_id = anno["super_category_id"]
            category_id = anno["category_id"]
            #import pdb;pdb.set_trace()
            if category_id not in num_list[super_category_id].keys():
                num_list[super_category_id][category_id] = 0
            num_list[super_category_id][category_id] += 1
            cat_list[super_category_id].append(category_id)

        class_weight = [[] for i in range(num_superclass)]
        sum_weight = [0 for i in range(num_superclass)]
        for i in range(len(num_list)):
            supclass_num_list = num_list[i]
            max_num = max(supclass_num_list)
            class_weight[i] = [max_num / i if i != 0 else 0 for i in supclass_num_list]
            sum_weight[i] = sum(class_weight[i])
        return class_weight, sum_weight

    def _get_class_dict(self):
        super_class_dict = dict()
        sub_class_dict = dict()
        for i, anno in enumerate(self.data):
            sup_cat_id = (
                anno["super_category_id"] if "super_category_id" in anno else anno["image_label"]
            )
            if not sup_cat_id in super_class_dict:
                super_class_dict[sup_cat_id] = []
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )
            if not cat_id in sub_class_dict:
                sub_class_dict[cat_id] = []

            super_class_dict[sup_cat_id].append(i)
            sub_class_dict[cat_id].append(i)

        return super_class_dict,sub_class_dict

    def get_num_super_classes(self):
        return self.num_super_classes

    def get_num_classes(self):
        return self.num_classes_per_super_classes

    def update_transform(self, input_size=None):
        normalize = TRANSFORMS["normalize"](cfg=self.cfg, input_size=input_size)
        transform_list = [transforms.ToPILImage()]
        transform_ops = (
            self.cfg.TRANSFORMS.TRAIN_TRANSFORMS
            if self.mode == "train"
            else self.cfg.TRANSFORMS.TEST_TRANSFORMS
        )
        for tran in transform_ops:
            transform_list.append(TRANSFORMS[tran](cfg=self.cfg, input_size=input_size))
        transform_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transform_list)


    def get_annotations(self):
        return self.all_info['annotations']

    def __len__(self):
        return len(self.data)

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "pillow open {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_trans_image(self, img_idx):
        now_info = self.data[img_idx]
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)[None, :, :, :]