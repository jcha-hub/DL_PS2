"""
SGD Optimizer.  (c) 2021 Georgia Tech

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

from ._base_optimizer import _BaseOptimizer

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from sgd.py!")

class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum
        self.cache = {} # dict to store v_weight_prev, v_bias_prev for each model

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):        #update momentum for weights
                #have values m.weight, m.bias, self.learning_rate, self.momentum, m.dw, d.db
                #cache v_weight_prev, v_bias_prev for v_weight, v_bias

                if m.dw is not None:
                    v_w_prev = self.cache.get(m, {}).get('v_w', 0)
                    v_w = self.momentum * v_w_prev - self.learning_rate * m.dw
                    m.weight = m.weight + v_w
                else:
                    v_w = 0

            if hasattr(m, 'bias'):          #update momentum for bias
                v_b_prev = self.cache.get(m, {}).get('v_b', 0)

                if m.db is not None:
                    v_b = self.momentum * v_b_prev - self.learning_rate * m.db
                    m.bias = m.bias + v_b
                else:
                    v_b = 0

        #store values in cache
        self.cache[m] = {'v_w': v_w, 'v_b': v_b}

