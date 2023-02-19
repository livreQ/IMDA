#!/usr/bin/env python
# -*- coding=utf8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd



class MLPNet(nn.Module):
    def __init__(self, configs):
        """
        MLP network with ReLU
        """

        super().__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                for i in range(self.num_hidden_layers)
            ]
        )
        self.final = nn.Linear(self.num_neurons[-1], configs["output_dim"])
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability
        self.process_final = configs["process_final"]

    def forward(self, x):

        for hidden in self.hiddens:
            x = F.relu(hidden(self.dropout(x)))
        if self.process_final:
            return F.relu(self.final(self.dropout(x)))
        else:
            # no dropout or transform
            return self.final(x)


class ConvNet(nn.Module):
    def __init__(self, configs):
        """
        Feature extractor for the image (digits) datasets
        """

        super().__init__()
        self.channels = configs["channels"]  # number of channels
        self.num_conv_layers = len(configs["conv_layers"])
        self.num_channels = [self.channels] + configs["conv_layers"]
        # Parameters of hidden, cpcpcp, feature learning component.
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(self.num_channels[i], self.num_channels[i + 1], kernel_size=3)
                for i in range(self.num_conv_layers)
            ]
        )
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability

    def forward(self, x):

        dropout = self.dropout
        for conv in self.convs:
            x = F.max_pool2d(F.relu(conv(dropout(x))), 2, 2, ceil_mode=True)
        x = x.view(x.size(0), -1)  # flatten
        return x


class MLPNet_digits(nn.Module):
    def __init__(self, configs):
        """
        MLP network with ReLU
        """

        super().__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                for i in range(self.num_hidden_layers)
            ]
        )
        self.final = nn.Linear(self.num_neurons[-1], configs["output_dim"])
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability
        self.process_final = configs["process_final"]

    def forward(self, x, latent=False):

        for hidden in self.hiddens:
            x = F.relu(hidden(self.dropout(x)))
        latent_x = x
        if self.process_final:
            if latent:
                return latent_x, F.relu(self.final(self.dropout(x)))
            else:
                return F.relu(self.final(self.dropout(x)))
        else:
            # no dropout or transform
            if latent:
                return latent_x, self.final(x)
            else:
                return self.final(x)
