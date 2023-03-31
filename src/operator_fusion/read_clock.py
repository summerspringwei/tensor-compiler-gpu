'''
This file parse latency from fused kernels and prints the layerwise latency.
'''
import torch
import numpy as np
import os


def read_profile_cycles(file_path, gpu_freqency, minimal_blocks):
  tensor_model = torch.jit.load(file_path)
  tensor = list(tensor_model.parameters())[0]
  print(tensor)
  [stages, blocks, warps] = tensor.shape
  print("stages:{}, blocks:{}, warps:{}".format(stages, blocks, warps))
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
  # a = np.array([13.8385, 14.0680, 11.8410,  4.5338,  9.3318,  2.5343,  4.4078, 26.1510, 31.7826,  4.1220,  4.4575])
  # b = a * (1.5)
  # print(b)
  # print(np.sum(b))
  # dir_path = "/home/xiachunwei/Projects/tensor-compiler-gpu/release/"
  # file_path = os.path.join(dir_path, "profile_clock.pt")
  # read_profile_cycles(file_path, 765, 108)
  # attn_file_path = os.path.join(dir_path, "attn_profile_clock.pt")
  # read_profile_cycles(attn_file_path, 1410, 72)
  # attn_file_path = os.path.join(dir_path, "feed_forward_profile_clock.pt")
  # read_profile_cycles(attn_file_path, 1410, 72)

  dir_path = "/home/xiachunwei/Projects/tensor-compiler-gpu/101_release/"
  attn_file_path = os.path.join(dir_path, "profile_clock.pt")
  read_profile_cycles(attn_file_path, 1410, 108)
