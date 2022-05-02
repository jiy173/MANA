#!/usr/bin/env python3

import torch
import torch.nn.functional as F


class lossfun(torch.nn.Module):
    def __init__(self):
        super(lossfun, self).__init__()

    def forward(self, gt, oup):
        loss = F.l1_loss(oup, gt)
        return loss

