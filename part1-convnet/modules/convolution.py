"""
2d Convolution Module.  (c) 2021 Georgia Tech

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
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, Cin, H, W)
        : filter  shape (Cout, Cin, k, k)
        : bias shape (Cout,)
        : output shape (N, Cout, H_out, W_out)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        #get values weights, bias, stride, padding, etc
        s = self.stride
        p = self.padding
        k = self.kernel_size        #kernel dimensions
        Cin = self.in_channels      # number of channels input, example 3 for RBG
        Cout = self.out_channels    # number of filters
        N = x.shape[0]              # number of samples

        w= self.weight
        b = self.bias

        #add padding to input x, calculate output sizes H_out, W_out prior to padding x
        H = x.shape[2]
        W = x.shape[3]
        H_out = (H + 2 * p - k) // s + 1  # need as int
        W_out = (W + 2 * p - k) // s + 1

        x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
        #calc new input sizes after padding
        H = x_pad.shape[2]
        W = x_pad.shape[3]

        def conv_single_step(slice, w, b):
            # slice from x: input data of shape (N, C, H, W) -> slice (Cin, f, k) with filter (Cin, f, f)

            # element wise product of 3D array of input slice and filter (not count number of samples or filters)
            s = np.multiply(slice, w)
            # sum over all entries in volume s
            Z = np.sum(s)
            # add bias, cast as float with one scalar b per filter
            b = np.squeeze(b)
            Z = Z + b
            return Z

        # initialize output vol Z
        Z = np.zeros((N, Cout, H_out, W_out))

        for n in range(N):
            x_n = x_pad[n]  # select one training sample from input x

            for hi in range(H_out):
                for wi in range(W_out):
                    for c in range(Cout):
                        # get slice dimensions
                        vert_start = hi * s
                        vert_end = vert_start + k
                        hort_start = wi * s
                        hort_end = hort_start + k

                        # get slice from single training example
                        window = x_n[:, vert_start:vert_end, hort_start:hort_end]

                        # convolve 3D slice with single 3D filter w and bias b to get scalar, loops later over num filters
                        weights = w[c, :, :, :]
                        biases = b[c]
                        Z[n, c, hi, wi] = conv_single_step(window, weights, biases)

        out = Z
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        #get info
        s = self.stride
        p = self.padding
        k = self.kernel_size  # kernel dimensions
        Cin = self.in_channels  # number of channels input, example 3 for RBG
        Cout = self.out_channels  # number of filters
        N = x.shape[0]  # number of samples

        w = self.weight
        b = self.bias

        # add padding to input x, calculate output sizes H_out, W_out prior to padding x
        H = x.shape[2]
        W = x.shape[3]
        H_out = (H + 2 * p - k) // s + 1  # need as int
        W_out = (W + 2 * p - k) // s + 1

        x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
        # calc new input sizes after padding
        H = x_pad.shape[2]
        W = x_pad.shape[3]

        #initialize dx, dw, db with zeros in correct shapes to match x, w, b
        dx = np.zeros_like(x)
        dx_pad = np.zeros_like(x_pad)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)

        for n in range(N):
            x_pad_n = x_pad[n]
            dx_pad_n = dx_pad[n]

            for hi in range(H_out):
                for wi in range(W_out):
                    for c in range(Cout):
                        # get slice dimensions
                        vert_start = hi * s
                        vert_end = vert_start + k
                        hort_start = wi * s
                        hort_end = hort_start + k

                        # get slice from single training example
                        window = x_pad_n[:, vert_start:vert_end, hort_start:hort_end]

                        #update gradients for this slice and filter
                        dx_pad_n[:, vert_start:vert_end, hort_start:hort_end] += w[c,:,:,:] * dout[n,c,hi,wi]
                        dw[c,:,:,:] += window * dout[n, c, hi, wi]
                        db[c] += dout[n, c, hi, wi]
            #convert from dx_pad to dx - trim away padding
            dx[n,:,:,:] = dx_pad_n[:, p:-p, p:-p]

        print("dx", dx.shape)

        self.dx = dx
        self.dw = dw
        self.db = db


