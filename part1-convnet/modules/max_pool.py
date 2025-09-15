"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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

import numpy as np
import math

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        N = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        H_out = math.floor((H - self.kernel_size)/self.stride + 1)
        W_out = math.floor((W - self.kernel_size)/self.stride + 1)

        #initialize output
        out = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for hi in range(H_out):
                    for wi in range(W_out):
                            out[n, c, hi, wi] = np.max(x[n, c, hi*self.stride:(hi*self.stride + self.kernel_size), wi*self.stride:(wi*self.stride + self.kernel_size)])


        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """x
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache

        #dL_dx = 1 only for index positions in input array x in which value was the max value in pooling operation
        out = self.forward(x)
        dx = np.zeros_like(x)
        # print('x shape:', x.shape)
        # print('out shape: ', out.shape)
        # print('dout shape: ', dout.shape)

        N = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        for n in range(N):
            for c in range(C):
                for hi in range(H_out):
                    for wi in range(W_out):
                            #gets 2D slice from 4D tensor input
                            slice = x[n, c, hi*self.stride:(hi*self.stride + self.kernel_size), wi*self.stride:(wi*self.stride + self.kernel_size)]
                            # for each slice, get the row, col index of the max element and set it to one to create the local gradient dL/dout
                            max_pos = np.argmax(slice)
                            max_pos = np.unravel_index(max_pos, slice.shape)

                            slice_max_r_idx = max_pos[0] + hi*self.stride   #height position
                            slice_max_c_idx  = max_pos[1]  + wi*self.stride    #width position
                            max_idx = (n, c, slice_max_r_idx, slice_max_c_idx)

                            #troubleshooting
                            # print("max idx ", max_idx)

                            #multiply local gradient by output gradient to get dx
                            dx[max_idx] = 1 * dout[n, c, hi, wi]           # can only use tuple to take slice, not list
        #get dx
        self.dx = dx
        # print("dx: ", self.dx)

        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
