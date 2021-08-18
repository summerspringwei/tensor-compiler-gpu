from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.autograd import Function

import _transform_preds as _backend


class _TransformPreds(Function):
    @staticmethod
    def forward(ctx, coords, center, scale, output_size):
        output = _backend.transform_preds_cuda_forward(coords, center, scale, output_size)
        return output


transform_preds = _TransformPreds.apply


class TransformPreds(nn.Module):
    def __init__(self):
        super().__init__(self)
    
    def forward(self, coords, center, scale, output_size):
        return transform_preds(coords, center, scale, output_size)
