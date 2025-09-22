"""
Focal Loss Wrapper.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from focal_loss.py!")


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return: tensor containing the weights for each class
    """

    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    per_cls_weights = None
    #per paper, E_n = effective number of examples, reweight by 1/E_n
    tensor_cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
    per_cls_weights = (1-beta)/(1-beta **tensor_cls_num_list)

    #normalize - output div by samples in batch and mult by number of classes, to keep weights close to normal CE
    per_cls_weights = per_cls_weights/per_cls_weights.sum() * len(cls_num_list)
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        #want softmax class balanced focal loss

        #1 - need to calc pt- for each sample its model's estimated prob for the true class for that sample, label
        #p is shape (N, C) were C is 1...C are number of classes)
        # targets - labels (C,)      use F.logSoftmax not F.softmax for more accuracy since F.cross_entropy uses logSoftmax for its calcs
        log_p = F.log_softmax(input, dim=1)
        N = target.shape[0]      #batch size
        log_pt = log_p[torch.arange(N), target]
        pt = log_p.exp()[torch.arange(N), target]

        #2 - get sample weights for each sample in batch
        sample_weights = self.weight[target]

        # print('printouts------------------------')
        # print('input: ', input.shape)
        # print('target: ', target.shape)
        # print('pt: ', pt.shape)
        # print("sample_weights: ", sample_weights.shape)


        loss = - sample_weights * (1-pt)**self.gamma * log_pt    # by default averaging over samples, reduction=mean()
        loss = loss.mean()       #test compares to F.cross_entropy, which averages with mean() by default. So output also needs to be a scalar
        # print("loss: ", loss.shape)

        ce_loss = F.cross_entropy(input, target, weight=self.weight.float())        #does class balance CE but not class balance focal loss like our custom one
        # print("ce_loss: ", ce_loss.shape)
        # print('ce_loss value: ', ce_loss )

        return loss
