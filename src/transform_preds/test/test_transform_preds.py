
import torch 
import transform_preds

def test_transform_preds():
    dets = torch.reshape(torch.tensor([78.93826, 163.65175, 78.93826, 163.65175, 78.93826, 163.65175], torch.device('cuda')), (1,1,6))
    center = torch.tensor([240., 320.])
    scale = torch.tensor([512., 672.])
    output_size = torch.tensor([128, 168])
    trans = transform_preds.TransformPred()
    output = trans(dets, center, scale, output_size)
    print(output)

if __name__ == "__main__":
    test_transform_preds()
