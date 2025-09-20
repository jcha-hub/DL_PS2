"""
Linear Module.  (c) 2021 Georgia Tech

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

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from linear.py!")

class Linear:
    """
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    """

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        """
        out = None
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        #flatten input into shape X (N, d1*d2*....*dn), W (d1*d2*....*dn, outputsize  where N = input_number
        #y = XW + b where y(N, output_size)

        x_flatten = x.reshape(x.shape[0], -1)
        out = np.dot(x_flatten, self.weight) + self.bias
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #cache both original x after after flattening
        self.cache = (x, x_flatten)
        return out

    def backward(self, dout):
        """
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache[0]
        x_flatten = self.cache[1]  # x reshaped
        #orig x
        w = self.weight
        b = self.bias

        out = np.dot(x_flatten, w) + b

        # #input shapes
        # print('x', x.shape)         #x(10, 6)
        # print('w', w.shape)         #w(6, 5)
        # print('b', b.shape)         #b(5, )
        # print('out', out.shape)     #out(10, 5)

        dout_dw = x_flatten.T
        dout_dx = w.T

        dw = np.dot(dout_dw, dout)                          # must be (6,5) -> (6,10) (10,5)
        dx = np.dot(dout, w.T).reshape(x.shape)       # must be (10,6) -> (10,5)  (6,5)
        #reshape dx to shape of original x

        db = np.sum(dout, axis=0)  #must be (5,1)  -> (10,5).T (10,1)

        # #gradient shapes
        # print('dout', dout.shape)               # dout(10, 5)
        #
        # print('dout_dw', dout_dw.shape)         #dw(6, 10)
        # print('dout_dx', dout_dx.shape)         #dx(6, 5)
        # print('dw', dw.shape)                   ##w(6, 5)
        # print('dx', dx.shape)                   #x(10, 6) -> reshape to shape of x
        # print('db', db.shape)                   #(1,5)

        self.dw = dw
        self.db = db
        self.dx = dx

