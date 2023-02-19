#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qi.chen.1@ulaval.ca
# Created Time : Tue May 10 22:48:42 2022
# File Name: src/main/IMDA.py
# Description:
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from utils.module import L2ProjFunction, GradientReversalLayer



class IMDABase(nn.Module):
    def __init__(self, args, configs):
        """
        Wasserstein Semi-supervised Multi-Source Transfer
        """
        super().__init__()
        self.num_src_domains = configs["num_src_domains"]
        self.num_class = configs["num_classes"]
        self.feature_dim = configs["feature_dim"]
        # Gradient reversal layer.
        self.grl = GradientReversalLayer.apply
        self.epsilon = args.epsilon
        self.tau = args.tau
        self.gp_coef = args.gp# gradient penalty Wasserstein
        self.dataset = args.name# digits/amazon reviews
        self.lr = args.lr
        self.temp = args.temp
        self.W1_sup_coef = args.W1_sup_coef#0.05
        self.W1_unsup_coef = args.W1_unsup_coef#1
        self.W1_discri_coef1 = args.W1_discri_coef1#0.1
        self.W1_discri_coef2 = args.W1_discri_coef2
        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
        # source domain weights
        self.alpha = torch.from_numpy(np.ones([self.num_src_domains]) / self.num_src_domains)
        self.L = 1
        self.M = 1
        self.K = 1
        # SGLD
        self.add_noise = False

    def cal_grad_norm(self, grad):
        para_norm = 0
        for g in grad:
            #para_norm += g.data.norm(2) ** 2
            para_norm += torch.sum(g ** 2) + 1e-10
            #print(para_norm)
        return para_norm

    def forward(self, s_xs, s_ys, t_x, t_y, t_x_prime=None):
        """
        :param s_xs:     A list of k inputs from k source domains.
        :param s_ys:    A list of k outputs from k source domains.
        :param t_x:     Labeled input from the target domain.
        :param t_y:     Few label of target domain
        :param t_x_prime: Unlabeled input from the target domain
        :return:            tuple(aggregated loss, domain weights)
        """

        # Compute features
        s_features = []

        for dom_idx in range(self.num_src_domains):
            s_features.append(self.feature_net(s_xs[dom_idx]))

        if self.tau != 0:
            t_features = self.feature_net(t_x)
        if self.tau != 1:
            t_features_prime = self.feature_net(t_x_prime)


        # Classification probabilities on k source domains
        logprobs = []
        for dom_idx in range(self.num_src_domains):
            logprobs.append(F.log_softmax(self.class_net(s_features[dom_idx]), dim=1))

        # source prediction losses
        src_cls_losses = torch.stack(
            [
                F.nll_loss(logprobs[dom_idx], s_ys[dom_idx]) for dom_idx in range(self.num_src_domains)
            ]
        )
        R_S_alpha = torch.sum(src_cls_losses * self.alpha)
        # target prediction loss
        if self.epsilon < 1 and self.epsilon > 0:
            R_T = F.nll_loss(F.log_softmax(self.class_net(t_features), dim=1), t_y)
        else:
            R_T = R_S_alpha - R_S_alpha

        #print("R_S_alpha:", R_S_alpha)
        #print("R_T:", R_T)
        # Wasserstein distance between Scal^\alpha and Tcal
        s_loss, t_loss, s_loss_pseudo, t_pseudo_loss = 0, 0, 0, 0
        batch_size = s_xs[0].shape[0]
        W1_sup, W1_unsup = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
        aux_src_cls_losses = []
        aux_src_cls_losses_pseudo = []
        s_weighted_features = torch.zeros_like(s_features[0])
        #s_weighted_inputs = torch.zeros_like(s_xs[0])
        for dom_idx in range(self.num_src_domains):
            # weighted src adversarial loss
            s_loss += self.alpha[dom_idx] * F.nll_loss(F.log_softmax(self.su_W_net(self.grl(s_features[dom_idx])), dim=1), s_ys[dom_idx])
            aux_src_cls_losses.append(F.nll_loss(F.log_softmax(self.su_W_net(self.grl(s_features[dom_idx])), dim=1), s_ys[dom_idx]))
            s_loss_pseudo += self.alpha[dom_idx] * F.nll_loss(F.log_softmax(self.un_W_net(s_features[dom_idx]), dim=1), s_ys[dom_idx])
            aux_src_cls_losses_pseudo.append(F.nll_loss(F.log_softmax(self.un_W_net(s_features[dom_idx]), dim=1), s_ys[dom_idx]))
            s_weighted_features += self.alpha[dom_idx] * s_features[dom_idx]
            #s_weighted_inputs += self.alpha[dom_idx] * s_xs[dom_idx]

        aux_src_cls_losses = torch.stack(aux_src_cls_losses)
        aux_src_cls_losses_pseudo = torch.stack(aux_src_cls_losses_pseudo)
        # W(\Tcal_U, \Scal_U^\alpha)
        flag = -1
        domain_loss = 0
        if self.tau != 0:
            #print("supervised transfer")
            t_loss = F.nll_loss(F.log_softmax(self.su_W_net(self.grl(t_features)),dim=1), t_y)
            # domain loss is the Wasserstein loss (current w.o gradient penality)
            #if t_loss > s_loss:
            supervised_domain_loss =  s_loss - t_loss # - W_1
            # else:
            #     flag = 1
            #     supervised_domain_loss =  t_loss - s_loss
            # Defining domain regularization loss (gradient penality)
            epsilon = np.random.rand()
            interpolated = epsilon * s_weighted_features + (1 - epsilon) * t_features
            inter_f = self.su_W_net(interpolated)
            # The following compute the penalty of the Lipschitz constant
            penalty_coefficient = 10.0
            # torch.norm can be unstable? https://github.com/pytorch/pytorch/issues/2534
            # f_gradient_norm = torch.norm(torch.autograd.grad(torch.sum(inter_f), interpolated)[0], dim=1)
            f_gradient = torch.autograd.grad(torch.sum(inter_f), interpolated, create_graph=True, retain_graph=True)[0]
            f_gradient_norm = torch.sqrt(torch.sum(f_gradient ** 2, dim=1) + 1e-10)
            #print(f_gradient_norm.shape)
            domain_gradient_penalty = penalty_coefficient * torch.mean((f_gradient_norm - self.L*self.M) ** 2)
            #print("Wassertein distance:", supervised_domain_loss)
            W1_sup = supervised_domain_loss + self.gp_coef * domain_gradient_penalty
            domain_loss = supervised_domain_loss
            #print("W:", W1_sup)

        #W(\Tcal_{U,V}, \Scal_U^\alpha)
        if self.tau != 1:
            #print("unsupervised transfer")
            t_pseudo_loss = self.W1_discri_coef1 * F.nll_loss(F.log_softmax(self.un_W_net(t_features_prime), dim=1),\
                                                             torch.argmax(F.log_softmax(self.class_net(t_features_prime), dim=1),1))\
                            + self.W1_discri_coef2 * F.nll_loss(F.log_softmax(self.grl(self.class_net(t_features_prime)), dim=1), \
                                torch.argmax(F.log_softmax(self.un_W_net(t_features_prime), dim=1),1))
            # domain loss is the Wasserstein loss (current w.o gradient penality)
            unsupervised_domain_loss =  s_loss_pseudo - t_pseudo_loss
            # Defining domain regularization loss (gradient penality)
            # t_x_prime.requires_grad = True
            # s_weighted_inputs.requires_grad = True
            #print(t_x_prime.requires_grad, s_weighted_inputs.requires_grad)
            epsilon = np.random.rand()
            # interpolated = epsilon * s_weighted_inputs + (1 - epsilon) * t_x_prime
            # inter_f = self.un_W_net(self.feature_net(interpolated))
            interpolated = epsilon * s_weighted_features + (1 - epsilon) * t_features_prime
            inter_f = self.un_W_net(interpolated)
            # The following compute the penalty of the Lipschitz constant
            penalty_coefficient = 10.0
            # torch.norm can be unstable? https://github.com/pytorch/pytorch/issues/2534
            # f_gradient_norm = torch.norm(torch.autograd.grad(torch.sum(inter_f), interpolated)[0], dim=1)
            f_gradient = torch.autograd.grad(torch.sum(inter_f), interpolated, create_graph=True, retain_graph=True)[0]
            f_gradient_norm = torch.sqrt(torch.sum(f_gradient ** 2, dim=1) + 1e-10)
            # f_gradient_norm = 0
            #print(f_gradient.shape)
            # for f_g in f_gradients:
            #     print(f_g.shape)
            #     f_gradient_norm += torch.sum(f_g ** 2, dim=1) + 1e-10
            # f_gradient_norm = torch.sqrt(f_gradient_norm)
            domain_gradient_penalty = penalty_coefficient * torch.mean((f_gradient_norm - self.L* self.M * self.K) ** 2)
            W1_unsup = unsupervised_domain_loss + self.gp_coef * domain_gradient_penalty
            domain_loss = unsupervised_domain_loss
            #print("\ns_loss_pseudo:", s_loss_pseudo.detach().cpu().numpy())
            #print("t_loss_pseudo:", t_pseudo_loss.detach().cpu().numpy())
        train_loss = self.tau * (1 - self.epsilon) * R_T \
                + self.tau * self.epsilon * R_S_alpha \
                + self.W1_sup_coef * self.tau * self.epsilon * W1_sup \
                + self.W1_unsup_coef *(1 - self.tau) * W1_unsup

        u_grads = torch.autograd.grad(train_loss, self.feature_net.parameters(),retain_graph=True)
        v_grads = torch.autograd.grad(train_loss, self.class_net.parameters(),retain_graph=True)
        # #print(u_grad.shape, v_grad.shape)
        u_g_norm = self.cal_grad_norm(u_grads)
        v_g_norm = self.cal_grad_norm(v_grads)
        # u_g_norm = torch.mean(torch.sqrt(torch.sum(u_grad ** 2, dim=1) + 1e-10)**2)
        # v_g_norm = torch.mean(torch.sqrt(torch.sum(v_grad ** 2, dim=1) + 1e-10)**2)

        convex_loss = (self.tau * self.epsilon * src_cls_losses \
                - 0.9 * self.tau * self.epsilon * aux_src_cls_losses \
                + self.W1_unsup_coef * (1 - self.tau) * (self.W1_discri_coef2 * src_cls_losses - aux_src_cls_losses_pseudo)).detach()

        return train_loss, convex_loss, R_T, R_S_alpha, W1_sup, W1_unsup, domain_loss , u_g_norm, v_g_norm

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)
        return F.log_softmax(x, dim=1)
