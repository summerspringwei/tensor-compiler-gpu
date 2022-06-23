from statistics import variance
import  bert_binding
import torch
import numpy as np


def test_fused_query_key_matmul_softmax():
  query = torch.ones((1*12, 128, 64), dtype=torch.half, device="cuda")
  key = torch.ones((1*12, 128, 64), dtype=torch.half, device="cuda")
  # print(bert_binding.d_sigmoid(a))
  output = bert_binding.fused_query_key_matmul_softmax(query, key)
  output_np = np.reshape(output.cpu().numpy(), (12, 128, 128))

  torch_output = query @ key.transpose(-2, -1)
  torch_output = torch.softmax(torch_output, -1, torch.float16)
  np.testing.assert_allclose(output.cpu().numpy(), torch_output.cpu().numpy())
  print(output.shape)
  for i in range(12):
    for j in range(128):
      for k in range(128):
        if output_np[i, j, k] != 0.0078125:
          print(i, j, k)
  # output_np = np.reshape(output.cpu().numpy(), (12, 128* 128))
  # np.savetxt("foo.csv", output_np, delimiter=',')


def init_tensor(m, n):
  arr = np.zeros((m, n))
  for i in range(m):
    for j in range(n):
      arr[i, j] = (i % 17) / 20 / 32
  return torch.tensor(arr, dtype=torch.half, device="cuda")

def test_fused_feedforward():
  
  # src = torch.rand(128, 768, dtype=torch.half, device="cuda") / 16
  # weight1 = torch.rand(3072, 768, dtype=torch.half, device="cuda") / 16
  # weight2 = torch.rand(768, 3072, dtype=torch.half, device="cuda") / 256

  # src = torch.ones((128, 768), dtype=torch.half, device="cuda") / 16
  # weight1 = torch.ones((3072, 768), dtype=torch.half, device="cuda") / 16
  # weight2 = torch.ones((768, 3072), dtype=torch.half, device="cuda") / 64
  src = init_tensor(128, 768)
  weight1 = init_tensor(3072, 768)
  weight2 = init_tensor(768, 3072)
  print(src.cpu().numpy())
  print(weight1.cpu().numpy())
  print(weight2.cpu().numpy())
  print("*"*20)
  output,sum, variance = bert_binding.fused_feed_forward(src, weight1, weight2)
  
  print(sum)
  print(variance)
  print(output)

  # Equal torch implementation
  output1 = torch.matmul(src, weight1.transpose(-2, -1))
  output2 = torch.matmul(output1, weight2.transpose(-2, -1))
  src = src+output2

  # t_avg = torch.sum(src, 1, dtype=torch.float32) / 768
  # t_delt = src - t_avg[:, None]
  # t_delt2 = t_delt * t_delt
  # t_variance = torch.sum(t_delt2, 1, dtype=torch.float32) / 768
  # src = (src - t_avg[:, None]) / torch.sqrt(t_variance + 0.00001)[:, None]
  src = torch.layer_norm(src, (768,))
  print("src")
  # print(t_avg)
  # print(t_variance)
  print(src)
  np.testing.assert_allclose(output.cpu().numpy(), src.cpu().numpy(), rtol=0.1)


if __name__=="__main__":
  # test_fused_query_key_matmul_softmax()
  test_fused_feedforward()
