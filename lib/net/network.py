import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import (res32_cifar,res32_cifar_group, res50,res50_group, res10, res10_group, res152,res152_group)
from modules import GAP, FCNorm, FCGroupNorm, Identity, SEN, GMP, LWS, LWS_bias
import copy
import numpy as np
import cv2

class Network(nn.Module):
    def __init__(self, cfg, groups, mode="train", num_classes=1000):
        super(Network, self).__init__()
        pretrain = (
            True
            if mode == "train"
            and cfg.RESUME_MODEL == ""
            and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )


        self.num_classes = num_classes
        self.cfg = cfg
        self.group = groups

        self.backbone = eval(self.cfg.BACKBONE.TYPE)(
            self.cfg,
            pretrain=pretrain,
            pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
            last_layer_stride=2,
        )
        self.module = self._get_module()
        self.classifier = self._get_classifer()


    def forward(self, x, **kwargs):
       # print(x[0].shape)
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "classifier_flag" in kwargs:
            return self.classifier(x)
        elif 'feature_maps_flag' in kwargs:
            return self.extract_feature_maps(x)
        elif 'layer' in kwargs and 'index' in kwargs:
            if kwargs['layer'] in ['layer1', 'layer2', 'layer3']:
                x = self.backbone.forward(x, index=kwargs['index'], layer=kwargs['layer'], coef=kwargs['coef'])
            else:
                x = self.backbone(x)
            x = self.module(x)
            if kwargs['layer'] == 'pool':
                x = kwargs['coef']*x+(1-kwargs['coef'])*x[kwargs['index']]
            x = x.view(x.shape[0], -1)
            x = self.classifier(x)
            if kwargs['layer'] == 'fc':
                x = kwargs['coef']*x + (1-kwargs['coef'])*x[kwargs['index']]
            return x

        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def get_backbone_layer_info(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            layers = 3
            blocks_info = [5, 5, 5]
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            layers = 4
            blocks_info = [1, 1, 1, 1]
        else:
            layers = 4
            blocks_info = [3, 4, 6, 3]
        return layers, blocks_info

    def extract_feature(self, x, **kwargs):
        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        return x

    def extract_feature_maps(self, x):
        x = self.backbone(x)
        return x

    def extract_feature_maps_multi(self, x):
        x = self.backbone(x)
        return x

    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False


    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone model has been loaded...")


    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            print(k)
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("All model has been loaded...")

    def get_fc(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        fc_weight_many = pretrain_dict['module.classifier_many.weight'].cpu().numpy()
        fc_bias_many = pretrain_dict['module.classifier_many.bias'].cpu().numpy()
        fc_weight_medium = pretrain_dict['module.classifier_medium.weight'].cpu().numpy()
        fc_bias_medium = pretrain_dict['module.classifier_medium.bias'].cpu().numpy()
        fc_weight_few = pretrain_dict['module.classifier_few.weight'].cpu().numpy()
        fc_bias_few = pretrain_dict['module.classifier_few.bias'].cpu().numpy()
        return [fc_weight_many, fc_weight_medium, fc_weight_few], [fc_bias_many, fc_bias_medium, fc_bias_few]


    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            num_features = 512
        else:
            num_features = 2048
        return num_features


    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "GMP":
            module = GMP()
        elif module_type == "Identity":
            module= Identity()
        elif module_type == "SEN":
            module= SEN(c=64)
        else:
            raise NotImplementedError

        return module


    def _get_classifer(self):
        bias_flag = self.cfg.CLASSIFIER.BIAS

        num_features = self.get_feature_length()
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE == "FC":
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        elif self.cfg.CLASSIFIER.TYPE == "FCGroupNorm":
            classifier = FCGroupNorm(num_features, self.num_classes, self.group)
        else:
            raise NotImplementedError

        return classifier


    def cam_params_reset(self):
        self.classifier_weights = np.squeeze(list(self.classifier.parameters())[0].detach().cpu().numpy())

    def get_CAM_with_groundtruth(self, image_idxs, dataset, size):
        ret_cam = []
        size_upsample = size
        for i in range(len(image_idxs)):
            idx = image_idxs[i]
            label = dataset.label_list[idx]
            self.eval()
            with torch.no_grad():
                img = dataset._get_trans_image(idx)
                feature_conv = self.forward(img.to('cuda'), feature_maps_flag=True).detach().cpu().numpy()
            b, c, h, w = feature_conv.shape
            assert b == 1
            feature_conv = feature_conv.reshape(c, h*w)
            cam = self.classifier_weights[label].dot(feature_conv)
            del img
            del feature_conv
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255*cam_img)
            ret_cam.append(cv2.resize(cam_img, size_upsample))
        return ret_cam

class Network_Group(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000):
        super(Network_Group, self).__init__()
        pretrain = (
            True
            if mode == "train"
            and cfg.RESUME_MODEL == ""
            and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.num_classes = num_classes
        self.cfg = cfg

        self.backbone = eval(self.cfg.BACKBONE.TYPE)(
            self.cfg,
            pretrain=pretrain,
            pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
            last_layer_stride=2,
        )
        self.module = self._get_module()
        #self.gate = self._get_gate()
        #self.classifier_many,self.classifier_medium,self.classifier_few,self.classifier_all = self._get_classifer()
        self.classifier_many, self.classifier_medium, self.classifier_few = self._get_classifer()

    def forward(self, x, **kwargs):
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "classifier_flag" in kwargs:
            x_few = self.classifier_few(x[0])
            x_medium = self.classifier_medium(x[1])
            x_many = self.classifier_many(x[2])
            x = [x_many, x_medium, x_few]
            return x
        elif 'feature_maps_flag' in kwargs:
            return self.extract_feature_maps(x)
        elif 'layer' in kwargs and 'index' in kwargs:
            if kwargs['layer'] in ['layer1', 'layer2', 'layer3']:
                x = self.backbone.forward(x, index=kwargs['index'], layer=kwargs['layer'], coef=kwargs['coef'])
            else:
                x = self.backbone(x)
            x = self.module(x)
            if kwargs['layer'] == 'pool':
                x = kwargs['coef']*x+(1-kwargs['coef'])*x[kwargs['index']]
            #x_all = self.classifier_many(x[3])
            x_many =self.classifier_many(x[2])
            x_medium = self.classifier_medium(x[1])
            x_few = self.classifier_few(x[0])
            x = [x_many, x_medium, x_few]
            if kwargs['layer'] == 'fc':
                x = kwargs['coef']*x + (1-kwargs['coef'])*x[kwargs['index']]
            return x
        x = self.backbone(x)
        x_out = []
        for branch in x:
            branch = self.module(branch)
            branch = branch.view(branch.shape[0], -1)
            x_out.append(branch)

        x_few = self.classifier_few(x_out[0])
        x_medium = self.classifier_medium(x_out[1])
        x_many = self.classifier_many(x_out[2])
        x = [x_many, x_medium, x_few]
        return x

    def get_backbone_layer_info(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            layers = 3
            blocks_info = [5, 5, 5]
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            layers = 4
            blocks_info = [1, 1, 1, 1]
        elif 'res50' in self.cfg.BACKBONE.TYPE:
            layers = 4
            blocks_info = [3, 4, 6, 3]
        else:
            layers = 4
            blocks_info = [3, 8, 36, 3]
        return layers, blocks_info

    def extract_feature(self, x, **kwargs):
        x = self.backbone(x)
        x_out = []
        for branch in x:
            branch = self.module(branch)
            branch = branch.view(branch.shape[0], -1)
            x_out.append(branch)
        return x_out

    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False


    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone model has been loaded...")


    def load_model(self, model_path):

        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            print(k)
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("All model has been loaded...")

    def get_fc(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            print(k)
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        #fc_weight_all = pretrain_dict['module.classifier_all.weight'].cpu().numpy()
       # fc_bias_all = pretrain_dict['module.classifier_all.bias'].cpu().numpy()
        fc_weight_many = pretrain_dict['module.classifier_many.fc.weight'].cpu().numpy()
        fc_bias_many = pretrain_dict['module.classifier_many.fc.bias'].cpu().numpy()
        fc_scales_many = pretrain_dict['module.classifier_many.scales'].cpu().numpy()
        fc_weight_medium = pretrain_dict['module.classifier_medium.fc.weight'].cpu().numpy()
        fc_bias_medium = pretrain_dict['module.classifier_medium.fc.bias'].cpu().numpy()
        fc_scales_medium = pretrain_dict['module.classifier_medium.scales'].cpu().numpy()
        fc_weight_few = pretrain_dict['module.classifier_few.fc.weight'].cpu().numpy()
        fc_bias_few = pretrain_dict['module.classifier_few.fc.bias'].cpu().numpy()
        fc_scales_few = pretrain_dict['module.classifier_few.scales'].cpu().numpy()
        return [fc_weight_many,fc_weight_medium,fc_weight_few ] ,[fc_bias_many,fc_bias_medium,fc_bias_few],[fc_scales_many,fc_scales_medium,fc_scales_few]#

    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            num_features = 512
        else:
            num_features = 2048
        return num_features


    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module= Identity()
        elif module_type == "SEN":
            module= SEN(c=64)
        else:
            raise NotImplementedError

        return module

    def _get_gate(self):
        gate = nn.Linear(64, 3, bias=True)
        return gate

    def _get_classifer(self):
        bias_flag = self.cfg.CLASSIFIER.BIAS
        num_features = self.get_feature_length()
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            classifier_many = FCNorm(num_features, self.num_classes)
            classifier_medium = FCNorm(num_features, self.num_classes)
            classifier_few = FCNorm(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE == "FC":
            classifier_many = nn.Linear(num_features, self.num_classes , bias=bias_flag)
            classifier_medium = nn.Linear(num_features, self.num_classes, bias=bias_flag)
            classifier_few = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        elif self.cfg.CLASSIFIER.TYPE == "LWS":
            classifier_many = LWS(num_features, self.num_classes, bias=bias_flag)
            classifier_medium = LWS(num_features, self.num_classes, bias=bias_flag)
            classifier_few = LWS(num_features, self.num_classes, bias=bias_flag)
        elif self.cfg.CLASSIFIER.TYPE == "LWS_bias":
            classifier_many = LWS_bias(num_features, self.num_classes, bias=bias_flag)
            classifier_medium = LWS_bias(num_features, self.num_classes, bias=bias_flag)
            classifier_few = LWS_bias(num_features, self.num_classes, bias=bias_flag)
        else:
            raise NotImplementedError

        #return classifier_many, classifier_medium, classifier_few, classifier_all
        return classifier_many, classifier_medium, classifier_few

    def _get_branch(self):
        num_features = self.get_feature_length()
        branch_many = SubGroup(num_features)
        branch_medium = SubGroup(num_features)
        branch_few = SubGroup(num_features)
        return branch_many, branch_medium, branch_few

    def cam_params_reset(self):
        self.classifier_weights = np.squeeze(list(self.classifier.parameters())[0].detach().cpu().numpy())

class SubGroup(nn.Module):
    def __init__(self,num_features):
        super(SubGroup, self).__init__()
        self.feat1 = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=1)
        self.feat2 = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=1)
        self.feat3 = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=1)

        #self.init_weights(self.feat1)
        #self.init_weights(self.feat2)
        #self.init_weights(self.feat3)

    def init_weights(self, m):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    def forward(self, x):
        x = self.feat1(x)
        x = self.feat2(x)
        x = self.feat3(x)
        return x