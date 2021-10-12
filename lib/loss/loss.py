import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import cv2

class CrossEntropy(nn.Module):
    def __init__(self, expert_idx, para_dict=None):
        super(CrossEntropy, self).__init__()

        self.para_dict = para_dict
        self.num_classes = self.para_dict["num_classes"]
        self.num_class_list = self.para_dict['num_class_list']
        self.device = self.para_dict['device']

        self.weight_list = None
        #setting about defferred re-balancing by re-weighting (DRW)
        self.drw = self.para_dict['cfg'].TRAIN.TWO_STAGE.DRW
        self.drw_start_epoch = self.para_dict['cfg'].TRAIN.TWO_STAGE.START_EPOCH #start from 1
    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        loss = F.cross_entropy(inputs, targets, weight=self.weight_list)

        return loss, inputs

    def update(self, epoch):
        """
        Adopt cost-sensitive cross-entropy as the default
        Args:
            epoch: int. starting from 1.
        """
        start = (epoch-1) // self.drw_start_epoch
        if start and self.drw:
            self.weight_list = torch.FloatTensor(np.array([min(self.num_class_list) / N for N in self.num_class_list])).to(self.device)


class ResCELoss(CrossEntropy):
    def __init__(self, expert_idx, para_dict=None):
        super(ResCELoss, self).__init__(expert_idx, para_dict)
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.num_classes = self.para_dict["num_classes"]
        self.num_class_list = self.para_dict['num_class_list']
        ranked = np.argsort(self.num_class_list)
        largest_indices = ranked[::-1].copy()
        self.many_index, self.medium_index, self.few_index = expert_idx

        self.celoss= nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()

    def one_hot(self, x, class_count):
        return torch.eye(class_count)[x, :]

    def forward(self, output, targets, **kwargs):

        [many_output_ori, medium_output_ori, few_output_ori,final_output] = output

        few_head_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        medium_head_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        many_head_loss = torch.tensor(0.0, requires_grad=True).to(many_output_ori.device)
        targets_cpu = targets.cpu().numpy()

        few_head_targets_cpu = [targets_cpu[j] for j in range(len(targets_cpu)) if targets_cpu[j] in self.few_index]
        few_head_idx_cpu = [j for j in range(len(targets_cpu)) if targets_cpu[j] in self.few_index]
        few_output = few_output_ori[few_head_idx_cpu, :]
        few_head_targets = torch.tensor(np.asarray(few_head_targets_cpu)).to(few_output.device)
        few_final_targets = final_output[few_head_idx_cpu, :].to(few_output.device)

        if len(few_head_targets):
            few_head_loss = self.celoss(few_output, few_head_targets)+\
                            torch.norm(few_output[:,np.concatenate((self.many_index,self.medium_index))], p=2)/len(np.concatenate((self.many_index,self.medium_index)))
        medium_head_targets_cpu = [targets_cpu[j] for j in range(len(targets_cpu)) if targets_cpu[j] not in self.many_index]
        medium_head_idx_cpu = [j for j in range(len(targets_cpu)) if targets_cpu[j] not in self.many_index]
        medium_output = medium_output_ori[medium_head_idx_cpu,:]
        medium_head_targets =  torch.tensor(np.asarray(medium_head_targets_cpu)).to(medium_output.device)

        if len(medium_head_targets):\
            medium_head_loss = self.celoss(medium_output, medium_head_targets)\
                             + torch.norm(medium_output[:, self.many_index], p=2)/len(self.many_index)

        many_head_targets = targets.to(many_output_ori.device)
        many_head_idx_cpu_final = [j  for j in range(len(targets_cpu)) if targets_cpu[j] in self.many_index]
        many_output = many_output_ori[many_head_idx_cpu_final, :]
        many_final_targets = final_output[many_head_idx_cpu_final, :].to(many_output.device)
        many_head_loss = self.celoss(many_output_ori, many_head_targets)

        return [many_head_loss, medium_head_loss, few_head_loss], final_output



class CrossEntropyLabelSmooth(CrossEntropy):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

        Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight of label smooth.
    """
    def __init__(self, para_dict=None):
        super(CrossEntropyLabelSmooth, self).__init__(para_dict)
        super(CrossEntropyLabelSmooth, self).__init__(para_dict)
        self.epsilon = self.para_dict['cfg'].LOSS.CrossEntropyLabelSmooth.EPSILON #hyper-parameter in label smooth
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class FocalLoss(CrossEntropy):
    r"""
    Reference:
    Li et al., Focal Loss for Dense Object Detection. ICCV 2017.
        Equation: Loss(x, class) = - (1-sigmoid(p^t))^gamma \log(p^t)
    Focal loss tries to make neural networks to pay more attentions on difficult samples.
    Args:
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
    """
    def __init__(self, para_dict=None):
        super(FocalLoss, self).__init__(para_dict)
        self.gamma = self.para_dict['cfg'].LOSS.FocalLoss.GAMMA #hyper-parameter
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        weight = (self.weight_list[targets]).to(targets.device) \
            if self.weight_list is not None else \
            torch.FloatTensor(torch.ones(targets.shape[0])).to(targets.device)
        label = get_one_hot(targets, self.num_classes)
        p = self.sigmoid(inputs)
        focal_weights = torch.pow((1-p)*label + p * (1-label), self.gamma)
        loss = F.binary_cross_entropy_with_logits(inputs, label, reduction = 'none') * focal_weights
        loss = (loss * weight.view(-1, 1)).sum() / inputs.shape[0]
        return loss, inputs

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = torch.FloatTensor(np.array([1 for _ in self.num_class_list])).to(self.device)
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = torch.FloatTensor(np.array([min(self.num_class_list) / N for N in self.num_class_list])).to(self.device)
            else:
                self.weight_list = torch.FloatTensor(np.array([1 for _ in self.num_class_list])).to(self.device)


def get_one_hot(label, num_classes):
    batch_size = label.shape[0]
    onehot_label = torch.zeros((batch_size, num_classes))
    onehot_label = onehot_label.scatter_(1, label.unsqueeze(1).detach().cpu(), 1)
    onehot_label = (onehot_label.type(torch.FloatTensor)).to(label.device)
    return onehot_label

class ClassBalanceCE(CrossEntropy):
    r"""
    Reference:
    Cui et al., Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.

        Equation: Loss(x, c) = \frac{1-\beta}{1-\beta^{n_c}} * CrossEntropy(x, c)

    Class-balanced loss considers the real volumes, named effective numbers, of each class, \
    rather than nominal numeber of images provided by original datasets.

    Args:
        beta(float, double) : hyper-parameter for class balanced loss to control the cost-sensitive weights.
    """
    def __init__(self, para_dict= None):
        super(ClassBalanceCE, self).__init__(para_dict)
        self.beta = self.para_dict['cfg'].LOSS.ClassBalanceCE.BETA
        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta ** N) for N in self.num_class_list])
        self.class_balanced_weight = torch.FloatTensor(self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes).to(self.device)

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = self.class_balanced_weight
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.class_balanced_weight




class ClassBalanceFocal(CrossEntropy):
    r"""
    Reference:
    Li et al., Focal Loss for Dense Object Detection. ICCV 2017.
    Cui et al., Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.

        Equation: Loss(x, class) = \frac{1-\beta}{1-\beta^{n_c}} * FocalLoss(x, c)

    Class-balanced loss considers the real volumes, named effective numbers, of each class, \
    rather than nominal numeber of images provided by original datasets.

    Args:
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                               putting more focus on hard, misclassiﬁed examples
        beta(float, double): hyper-parameter for class balanced loss to control the cost-sensitive weights.
    """
    def __init__(self, para_dict=None):
        super(ClassBalanceFocal, self).__init__(para_dict)
        self.beta = self.para_dict['cfg'].LOSS.ClassBalanceFocal.BETA
        self.gamma = self.para_dict['cfg'].LOSS.ClassBalanceFocal.GAMMA
        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta ** N) for N in self.num_class_list])
        self.class_balanced_weight = torch.FloatTensor(self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes).to(self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        import pdb;pdb.set_trace()
        inputs = inputs[0]
        weight = (self.weight_list[targets]).to(targets.device)
        label = get_one_hot(targets, self.num_classes)
        p = self.sigmoid(inputs)
        focal_weights = torch.pow((1-p)*label + p * (1-label), self.gamma)

        loss = F.binary_cross_entropy_with_logits(inputs, label, reduction = 'none') * focal_weights
        loss = (loss * weight.view(-1, 1)).sum() / inputs.shape[0]
        return [loss,loss,loss], inputs

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = self.class_balanced_weight
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.class_balanced_weight


class CostSensitiveCE(CrossEntropy):
    r"""
        Equation: Loss(z, c) = - (\frac{N_min}{N_c})^\gamma * CrossEntropy(z, c),
        where gamma is a hyper-parameter to control the weights,
            N_min is the number of images in the smallest class,
            and N_c is the number of images in the class c.

    The representative re-weighting methods, which assigns class-dependent weights to the loss function

    Args:
        gamma (float or double): to control the loss weights: (N_min/N_i)^gamma
    """
    def __init__(self, para_dict=None):
        super(CostSensitiveCE, self).__init__(para_dict)
        gamma = self.para_dict['cfg'].LOSS.CostSensitiveCE.GAMMA
        self.csce_weight = torch.FloatTensor(np.array([(min(self.num_class_list) / N)**gamma for N in self.num_class_list])).to(self.device)

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = self.csce_weight
        else:
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.csce_weight

class LDAMLoss(CrossEntropy):
    """
    LDAMLoss is modified from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).

    References:
    Cao et al., Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. NeurIPS 2019.

    Args:
        scale(float, double) : the scale of logits, according to the official codes.
        max_margin(float, double): margin on loss functions. See original paper's Equation (12) and (13)

    Notes: There are two hyper-parameters of LDAMLoss codes provided by official codes,
          but the authors only provided the settings on long-tailed CIFAR.
          Settings on other datasets are not avaliable (https://github.com/kaidic/LDAM-DRW/issues/5).
    """
    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__(para_dict)
        s = self.para_dict['cfg'].LOSS.LDAMLoss.SCALE
        max_m = self.para_dict['cfg'].LOSS.LDAMLoss.MAX_MARGIN
        self.max_m = max_m
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        #betas to control the **class-balanced loss (CVPR 2019)** weights according to LDAMLoss official codes
        if self.drw:
            self.betas = [0, 0.9999]
        else:
            self.betas = [0, 0]

    def update(self, epoch):
        idx = 1 if epoch >= self.drw_start_epoch else 0
        per_cls_weights = (1.0 - self.betas[idx]) / (1.0 - np.power(self.betas[idx], self.num_class_list))
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight_list = torch.FloatTensor(per_cls_weights).to(self.device)
    def forward(self, inputs, targets, **kwargs):
        index = torch.zeros_like(inputs, dtype=torch.uint8)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor)
        index_float = index_float.to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = inputs - batch_m
        outputs = torch.where(index, x_m, inputs)
        return F.cross_entropy(self.s * outputs, targets, weight=self.weight_list)

class SEQL(CrossEntropy):
    r"""
    Reference:
    Tan et al., Equalization Loss for Long-Tailed Object Recognition. CVPR 2020.

    SEQL means softmax equalization loss

    Equation:
        Eq.1: loss = - sum_j (y_j * log p_j),
        Eq.2: p_j = exp(z_j) / sum_k(w_k * exp(z_k),
        Eq.3: w_k = 1 - beta * T_lambda(n_k) * (1 - y_k),
        where y: one-hot label
              beta: a random variable with a probability of gamma (hyper-parameter) to be 1 and 1-gamma to be 0,
              n_k: the number of images in class k
              T_lambda: a threshold function which outputs 1 when n_k < lambda (hyper-parameter), otherwise, 0

    Args:
        gamma (float or double): a probability in beta, which is used to maintain the gradient of negative samples
        lambda (float or double): a threshold in T_lambda function

        For detailed ablation study of SEQL, see original paper's Table 7.
    """
    def __init__(self, para_dict=None):
        super(SEQL, self).__init__(para_dict)
        num_class_list = torch.FloatTensor(self.num_class_list)
        #print(num_class_list/num_class_list.sum())
        self.t_lambda = ((num_class_list/num_class_list.sum()) < self.para_dict['cfg'].LOSS.SEQL.LAMBDA).to(self.device)
        self.gamma = self.para_dict['cfg'].LOSS.SEQL.GAMMA

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        onehot_label = get_one_hot(targets, self.num_classes)
        #Eq.3
        beta = (torch.rand(onehot_label.size()) < self.gamma).to(self.device)
        w = 1 - beta * self.t_lambda * (1 - onehot_label)
        #Eq.2
        p = torch.exp(inputs) / ((w*torch.exp(inputs)).sum(axis=1, keepdims=True))
        #Eq.1
        loss = F.nll_loss(torch.log(p), targets)
        return loss


class BalancedSoftmaxCE(CrossEntropy):
    r"""
    References:
    Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.

    Equation: Loss(x, c) = -log(\frac{n_c*exp(x)}{sum_i(n_i*exp(i)})
    """

    def __init__(self, para_dict=None):
        super(BalancedSoftmaxCE, self).__init__(para_dict)
        self.bsce_weight = torch.FloatTensor(self.num_class_list).to(self.device)

    def forward(self, inputs, targets,  **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        logits = inputs + self.weight_list.unsqueeze(0).expand(inputs.shape[0], -1).log()
        loss = F.cross_entropy(input=logits, target=targets)
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int
        """
        if not self.drw:
            self.weight_list = self.bsce_weight
        else:
            self.weight_list = torch.ones(self.bsce_weight.shape).to(self.device)
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.bsce_weight
        # print(self.weight_list)
        # print(self.num_class_list)
        # print(self.num_classes)
        # print(self.drw)

class CDT(CrossEntropy):
    r"""
    References:
    Class-Dependent Temperatures (CDT) Loss, Ye et al., Identifying and Compensating for Feature Deviation in Imbalanced Deep Learning, arXiv 2020.

    Equation:  Loss(x, c) = - log(\frac{exp(x_c / a_c)}{sum_i(exp(x_i / a_i))}), and a_j = (N_max/n_j)^\gamma,
                where gamma is a hyper-parameter, N_max is the number of images in the largest class,
                and n_j is the number of image in class j.
    Args:
        gamma (float or double): to control the punishment to feature deviation.  For CIFAR-10, γ ∈ [0.0, 0.4]. For CIFAR-100
        and Tiny-ImageNet, γ ∈ [0.0, 0.2]. For iNaturalist, γ ∈ [0.0, 0.1]. We then select γ from several
        uniformly placed grid values in the range
    """
    def __init__(self, para_dict=None):
        super(CDT, self).__init__(para_dict)
        self.gamma = self.para_dict['cfg'].LOSS.CDT.GAMMA
        self.cdt_weight = torch.FloatTensor([(max(self.num_class_list) / i) ** self.gamma for i in self.num_class_list]).to(self.device)

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        inputs = inputs / self.weight_list
        loss = F.cross_entropy(inputs, targets)
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int
        """
        if not self.drw:
            self.weight_list = self.cdt_weight
        else:
            self.weight_list = torch.ones(self.cdt_weight.shape).to(self.device)
            start = (epoch-1) // self.drw_start_epoch
            if start:
                self.weight_list = self.cdt_weight


if __name__ == '__main__':
    pass