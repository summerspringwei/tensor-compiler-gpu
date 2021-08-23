from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.autograd import Function

import _transform_preds as _backend


# Replace for get_affine_transform in original CenterNet post process
class _GetAffineTransform(Function):
    @staticmethod
    def forward(ctx, coords, center, scale, output_size, shift, inv):
        output = _backend.get_affine_transform_forward(coords, center, scale, output_size, shift, inv)
        return output


get_affine_transform = _GetAffineTransform.apply


class GetAffineTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, coords, center, scale, output_size,  shift, inv):
        return get_affine_transform(coords, center, scale, output_size,  shift, inv)


# Replace for affine_transform in original CenterNet post process
class _AffineTransformDets(Function):
    @staticmethod
    def forward(ctx, dets, trans, num_classes, scale=1):
        output = _backend.affine_transform_dets_forward(dets, trans, num_classes, scale)
        return output


affine_transform_dets = _AffineTransformDets.apply


class AffineTransformDets(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, dets, trans, num_classes, scale=1):
        return affine_transform_dets(dets, trans, num_classes, scale)


