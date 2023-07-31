
import os
# We use this script to replace 
# threadIdx.x to threadIdx_x and blockIdx.x to blockIdx_x etc..
# to handle different grid/block dims between kernels
def rename_thread_binding(file_name):
  old_file_name = file_name+".tmp"
  os.rename(file_name, old_file_name)
  f = open(old_file_name, 'r')
  lines = f.readlines()
  old_strs = ["blockIdx.x", "blockIdx.y", "blockIdx.z", \
    "threadIdx.x", "threadIdx.y", "threadIdx.z"]
  new_strs = ["blockIdx_x", "blockIdx_y", "blockIdx_z", \
    "threadIdx_x", "threadIdx_y", "threadIdx_z"]
  new_lines = []
  for line in lines:
    for old_str, new_str in zip(old_strs, new_strs):
      line = line.replace(old_str, new_str)
    new_lines.append(line)
  
  f_new = open(file_name, 'w')
  f_new.writelines(new_lines)
  f_new.flush()
  f_new.close()
  f.flush()
  f.close()
  


if __name__=="__main__":
  # rename_thread_binding("kernels/bert_qkv_matmul_transpose.h")
  # rename_thread_binding("kernels/bert_query_key_matmul_softmax.h")
  # rename_thread_binding("kernels/bert_attn_value.h")
  rename_thread_binding("kernels/bert_attn_fc.h")



  