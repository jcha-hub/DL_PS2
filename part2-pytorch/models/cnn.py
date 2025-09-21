"""
Vanilla CNN model.  (c) 2021 Georgia Tech

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
    print("Roger that from cnn.py!")


class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and no padding                           #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        #RGB image so in_channels = 3
        #model is conv, ReLU, max pooling, and fully connected layer for classifcation
        self.conv = nn.Conv2d(in_channels=3,
                              out_channels=32,
                              kernel_size=7)     #stride=1, padding=0 are default
        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2)      #default stride is same as kernel_size

        self.fc = nn.Linear(in_features=5408,         #calculated from fc_in_features below
                            out_features = 10)


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = self.pool(F.relu(self.conv(x)))

        # #calculate the output size going into fully connected layer - comment out fc and run
        # #x.numel() = counts all elements in the tensor, i.e., batch_size * num_channels * height * width.
        # # x.shape[00 = batch size, and floor division results in num_channels * height * width
        # fc_in_features = x.numel() // x.shape[0]
        # print('hello-----------------------')
        # print('fc_in_features', fc_in_features)       #5408

        #need to flatten all dimensions except batch prior to input into fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)

        outs = x
        return outs
