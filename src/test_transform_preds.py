
import torch 
import numpy as np
from transform_preds import AffineTransformDets, GetAffineTransform


setup_str = """
import torch 
from transform_preds import AffineTransformDets, GetAffineTransform
det_list = []
for i in range(100):
    det_list.append([78.93826, 163.65175, 78.93826,\
        163.65175, 78.93826, 163.65175])
dets = torch.reshape(torch.tensor(det_list, device=torch.device('cuda')), [1, 100, 6])
trans = torch.reshape(torch.tensor([4., -0., -16.,\
        -0., 4., -16.], device=torch.device('cuda')), [1, 1, 6])


def test_affine_transform_dets(dets, trans):
    affine_func = AffineTransformDets()
    target_dets = affine_func(dets, trans, 80)
    

def test_get_affine_transform():
    center = torch.tensor([240., 320.])
    scale = torch.tensor([512., 672.])
    output_size = torch.tensor([128., 168.])
    shift = torch.Tensor([0., 0.])
    inv = 1
    get_affine_func = GetAffineTransform()
    trans = get_affine_func(center, scale, 0, output_size, shift, inv)

def bench_transform_preds(dets, trans):
    test_get_affine_transform()
    test_affine_transform_dets(dets, trans)
"""

run_str = """
bench_transform_preds(dets, trans)
"""

def run_timeit():
    import timeit
    print(timeit.timeit(run_str, setup=setup_str, number=1000))


if __name__ == "__main__":
    run_timeit()
