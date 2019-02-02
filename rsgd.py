#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch as th
from torch.optim.optimizer import Optimizer, required

spten_t = th.sparse.FloatTensor


def poincare_grad(p, d_p):
    r"""
    Function to compute Riemannian gradient from the
    Euclidean gradient in the Poincaré ball.

    Args:
        p (Tensor): Current point in the ball
        d_p (Tensor): Euclidean gradient at p
    """
    
    if d_p.is_sparse:
        # This part seem like it's removing dimensions with size 1.
        # Makes sense for dealing with sparse matrices.
        p_sqnorm = th.sum(
            p.data[d_p._indices()[0].squeeze()] ** 2, dim=1,
            keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        # p_sqnorm represents $\|\theta_{t}\|^{2}$ inside equation (5) of the paper.
        p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
        
        # d_p is updated to change Euclidean gradient $\nabla_{E}$ to Poincare
        # gradient $\nabla_{R}=g_{\theta}^{-1}\nabla_{E}$.
        # where $g_{\theta}$ is the Poincare ball metric tensor. 
        # [See the equation in beginning section 3 for calculating g_{\theta}]
        
        # This is shown in the update term in equation (5):
        # $\nabla_{R}=\frac{(1-p_sqnorm)^{2}}{4}\nabla_{E}$
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def euclidean_grad(p, d_p):
    # for Euclidean gradient, just return the gradient directly 
    # (Since we've defined d_p to be the Euclidean gradient)
    return d_p


def euclidean_retraction(p, d_p, lr):
    # This is p += -lr*dp, which is equivalent to $\theta_{t+1} = \theta_{t}-\eta_{t}\nabla_{R}$.
    p.data.add_(-lr, d_p)


class RiemannianSGD(Optimizer):
    r"""Riemannian stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rgrad (Function): Function to compute the Riemannian gradient from
            an Euclidean gradient
        retraction (Function): Function to update the parameters via a
            retraction of the Riemannian gradient
        lr (float): learning rate
    """

    def __init__(self, params, lr=required, rgrad=required, retraction=required):
        defaults = dict(lr=lr, rgrad=rgrad, retraction=retraction)
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            # consider each parameter
            for p in group['params']:
                # ignore parameters with no gradient request.
                if p.grad is None:
                    continue
                
                # If parameter has gradient, load it into the Euclidean gradient
                d_p = p.grad.data
                
                # Load the default learning rate $eta_{t}$ if no learning rate is specified.
                if lr is None:
                    lr = group['lr']
                    
                # Calculate the Riemannian gradient from the Euclidean gradient.
                # For Poincare ball, should call poincare_grad above.
                d_p = group['rgrad'](p, d_p)
                
                # Calculate the Euclidean retraction 
                group['retraction'](p, d_p, lr)
                
                # Seems that the projection part in equation (5) is not here.
        return loss
