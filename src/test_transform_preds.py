
import torch 
import numpy as np
import cv2
from transform_preds.transform_preds import TransformPreds, AffineTransformDets, GetAffineTransform

def test_transform_preds():
    dets = torch.reshape(torch.tensor([78.93826, 163.65175, 78.93826,\
         163.65175, 78.93826, 163.65175], device=torch.device('cuda')), [6])
    center = torch.tensor([240., 320.])
    scale = torch.tensor([512., 672.])
    output_size = torch.tensor([128., 168.])
    trans = TransformPreds()
    output = trans(dets, center, scale, output_size, 80)
    # print(output.to('cpu'))

def test_affine_transform_dets():
    dets = torch.reshape(torch.tensor([78.93826, 163.65175, 78.93826,\
         163.65175, 78.93826, 163.65175], device=torch.device('cuda')), [1, 1, 6])
    trans = torch.reshape(torch.tensor([4., -0., -16.,\
         -0., 4., -16.], device=torch.device('cuda')), [1, 1, 6])
    affine_func = AffineTransformDets()
    target_dets = affine_func(dets, trans, 80)
    print(target_dets.to('cpu'))

def test_get_affine_transform():
    center = torch.tensor([240., 320.])
    scale = torch.tensor([512., 672.])
    output_size = torch.tensor([128., 168.])
    shift = torch.Tensor([0., 0.])
    inv = 1
    get_affine_func = GetAffineTransform()
    trans = get_affine_func(center, scale, 0, output_size, shift, inv)
    print(trans)

def test_cv_get_affine_transform():
    src = np.array(
        [[240.000000, 240.000000, -16.000000, 0.000000, 0.000000, 0.000000],
        [320.000000, 64.000000, 64.000000, 0.000000, 0.000000, 0.000000, ],
        [1.000000, 1.000000, 1.000000, 0.000000, 0.000000, 0.000000, ],
        [0.000000, 0.000000, 0.000000, 240.000000, 240.000000, -16.000000, ],
        [0.000000, 0.000000, 0.000000, 320.000000, 64.000000, 64.000000, ],
        [0.000000, 0.000000, 0.000000, 1.000000, 1.000000, 1.000000]])
    src = np.array(
        [[240.000000, 320.000000, 1.000000, 0.000000, 0.000000, 0.000000, ],
        [0.000000, 0.000000, 0.000000, 240.000000, 320.000000, 1.000000, ],
        [240.000000, 64.000000, 1.000000, 0.000000, 0.000000, 0.000000, ],
        [0.000000, 0.000000, 0.000000, 240.000000, 64.000000, 1.000000, ],
        [-16.000000, 64.000000, 1.000000, 0.000000, 0.000000, 0.000000, ],
        [0.000000, 0.000000, 0.000000, -16.000000, 64.000000, 1.000000, ],])
    src = np.array(
        [[240.000000, 320.000000, 1.000000, 0.000000, 0.000000, 0.000000, ],
        [240.000000, 64.000000, 1.000000, 0.000000, 0.000000, 0.000000, ],
        [-16.000000, 64.000000, 1.000000, 0.000000, 0.000000, 0.000000, ],
        [0.000000, 0.000000, 0.000000, 240.000000, 320.000000, 1.000000, ],
        [0.000000, 0.000000, 0.000000, 240.000000, 64.000000, 1.000000, ],
        [0.000000, 0.000000, 0.000000, -16.000000, 64.000000, 1.000000, ]])
    dst = np.array([64, 64, 0, 84, 20, 20])
    # dst = np.array([64.000000, 84.000000, 64.000000, 20.000000, 0.000000, 20.000000])
    # trans = cv2.getAffineTransform(src, dst)
    trans = np.linalg.solve(src, dst)
    print(trans)

if __name__ == "__main__":
    # test_transform_preds()
    # test_affine_transform_dets()
    test_get_affine_transform()
    test_cv_get_affine_transform()
