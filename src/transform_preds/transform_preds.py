from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.autograd import Function

import _transform_preds as _backend


class _TransformPreds(Function):
    @staticmethod
    def forward(ctx, coords, center, scale, output_size, num_classes):
        output = _backend.transform_preds_forward(coords, center, scale, output_size, num_classes)
        return output


transform_preds = _TransformPreds.apply


class TransformPredsV1(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, coords, center, scale, output_size, num_classes):
        return transform_preds(coords, center, scale, output_size, num_classes)


class TransformPreds(TransformPredsV1):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, coords, center, scale, output_size, num_classes):
        return transform_preds(coords, center, scale, output_size, num_classes)