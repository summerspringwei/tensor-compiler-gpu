import torch

import efficientnet_se_module_binding

def test_se_module(batch_size, height, width, in_channel, reduce_channel):
  input_tensor = torch.ones((batch_size, in_channel, height, width), dtype=torch.float32, device="cuda").to("cuda")
  # se_reduce_weight = torch.ones((in_channel, reduce_channel), dtype=torch.float32, device="cuda").to("cuda")
  # se_expand_weight = torch.ones((reduce_channel, in_channel), dtype=torch.float32, device="cuda").to("cuda")
  se_reduce_weight = torch.ones((reduce_channel, in_channel), dtype=torch.float32, device="cuda").to("cuda")
  se_expand_weight = torch.ones((in_channel, reduce_channel), dtype=torch.float32, device="cuda").to("cuda")
  se_short_cut_add = efficientnet_se_module_binding.torch_dispatch_efficientnet_se_module_v2_short_cut_fused(input_tensor, se_reduce_weight, se_expand_weight)
  print(se_short_cut_add)


if __name__ == "__main__":
  test_se_module(1, 112, 112, 32, 8)
