# encoding: utf-8
# Author    : damionfan@163.com
# Datetime  : 2020/2/5 11:36
# User      : Damion Fan
# Product   : PyCharm
# Project   : gcn
# File      : model.py
# explain   : 文件说明

import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.module import Module


class GCN(Module):
    def __init__(self, in_count, out_count, bias=True):
        super(GCN, self).__init__()
        self.in_count = in_count
        self.out_count = out_count
        self.weight = Parameter(torch.FloatTensor(in_count, out_count))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_count))
        else:
            self.register_parameter('bias', None)
        self.rest_parameters()

    def rest_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

