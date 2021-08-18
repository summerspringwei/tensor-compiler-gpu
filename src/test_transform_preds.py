
import torch 
from transform_preds.transform_preds import TransformPreds

def test_transform_preds():
    dets = torch.reshape(torch.tensor([78.93826, 163.65175, 78.93826,\
         163.65175, 78.93826, 163.65175], device=torch.device('cuda')), (1,1,6))
    center = torch.tensor([240., 320.])
    scale = torch.tensor([512., 672.])
    output_size = torch.tensor([128., 168.])
    trans = TransformPreds()
    output = trans(dets, center, scale, output_size, 80)
    print(output.cpu())

if __name__ == "__main__":
    test_transform_preds()
