import torch

# out = torch.load("gpt2-torch-data/attn_c_proj.pt")
out = torch.load("gpt2-torch-data/MLP_c_fc.pt")
out2 = torch.permute(out, (1,0))
print(out)
print(out.shape)
print(out2)
print(out2.shape)