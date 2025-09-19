"""
CovNet Module.  (c) 2021 Georgia Tech

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

from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear
import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from conv_classifier.py!")

class ConvNet:
    """
    Max Pooling of input
    """
    def __init__(self, modules, criterion):
        self.modules = []
        self.cache = None


        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        """
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        """
        probs = None
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement forward pass of the model                                 #
        #############################################################################
        #add missing function for softmax
        def softmax(scores):
            """
            Compute softmax scores given the raw output from the model
            :param scores: raw scores from the model (N, num_classes)
            :return: prob: softmax probabilities (N, num_classes)
            """
            scores = scores - np.max(scores, axis=1, keepdims=True)
            prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # sum by row, keepdims for shape
            return prob

        out_prev = x      #initialize , at first iteration use input x
        for m in self.modules:
            out = m.forward(out_prev)
            out_prev = out

        #calculate probabilities and cross entropy loss, final output of model is out
        probs = softmax(out)

        softmaxCE = SoftmaxCrossEntropy()
        loss = softmaxCE.forward(x, y)

        #store x, y for use with backprop
        self.cache = (x,y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return probs, loss

    def backward(self):
        """
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        """
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement backward pass of the model                                #
        #############################################################################
        #get model input, output
        x, y = self.cache

        #get dout to start backprop
        softmaxCE = SoftmaxCrossEntropy()
        loss = softmaxCE.forward(x, y)

        #initialize dout with final model output, then during iterations becomes the upstream gradient of module in loop
        dout = softmaxCE.backward()

        for m in self.modules:
            m.backward(dout)
            dout = m.dx

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
