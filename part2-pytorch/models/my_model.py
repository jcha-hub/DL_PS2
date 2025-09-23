"""
MyModel model.  (c) 2021 Georgia Tech

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
    print("Roger that from my_model.py!")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #32- 30-15-13-13-6-4
        self.conv1 = nn.Conv2d(3,96,7, 1, padding=2)
        self.conv2 = nn.Conv2d(96, 96, 5, 1, padding=1)
        self.conv3 = nn.Conv2d(96, 128, 5, 1, padding='same')
        self.conv4 = nn.Conv2d(128,164,3, 1, padding='same')
        self.conv5 = nn.Conv2d(164, 196, 3, 1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features = 3136, out_features = 500)
        self.fc2 = nn.Linear(500, 80)
        self.fc3 = nn.Linear(80,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.dropout(x, 0.25)

        # # # fc_in_features = x.numel() // x.shape[0] - comment out fc to get input into fc
        # print('hello-----------------------')
        # fc_in_features = x.numel() // x.shape[0]
        # print('fc_in_features', fc_in_features)       #3136

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.25)   #reduce overfitting

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        outs = x
        return outs
