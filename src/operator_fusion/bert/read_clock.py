'''
This file parse latency from fused kernels and prints the layerwise latency.
'''
import torch
import numpy as np


def read_profile_cycles(file_path, gpu_freqency, minimal_blocks):
  tensor_model = torch.jit.load(file_path)
  tensor = list(tensor_model.parameters())[0]
  [stages, blocks, warps] = tensor.shape
  tensor = tensor[:, 0:minimal_blocks, :]
  cycles_stage = torch.zeros((stages-1, minimal_blocks, warps), dtype=torch.int64)
  latency_stage = torch.zeros((stages-1, minimal_blocks), dtype=torch.float32)
  for i in range(stages-1):
    cycles_stage[i] = (tensor[i+1]-tensor[i]).cpu()
    print(cycles_stage[i])
    latency_stage[i] = torch.mean((cycles_stage[i] / gpu_freqency).to(torch.float32), 1)
  
  layer_latency = torch.mean(latency_stage, 1)
  print(layer_latency)
  print(torch.sum(layer_latency))


if __name__=="__main__":
  file_path = "/home/xiachunwei/Projects/tensor-compiler-gpu/release/profile_clock.pt"
  read_profile_cycles(file_path, 1410, 72)
