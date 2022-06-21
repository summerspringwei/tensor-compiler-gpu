import  bert_binding
import torch
import numpy as np

query = torch.ones((1*12, 128, 64), dtype=torch.half, device="cuda")
key = torch.ones((1*12, 128, 64), dtype=torch.half, device="cuda")
# print(bert_binding.d_sigmoid(a))
output = bert_binding.fused_query_key_matmul_softmax(query, key)
output_np = np.reshape(output.cpu().numpy(), (12, 128, 128))
print(output.shape)
for i in range(12):
  for j in range(128):
    for k in range(128):
      if output_np[i, j, k] != 0.0078125:
        print(i, j, k)
        exit(0)
# output_np = np.reshape(output.cpu().numpy(), (12, 128* 128))
# np.savetxt("foo.csv", output_np, delimiter=',')
