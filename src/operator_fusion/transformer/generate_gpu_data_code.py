

def generate_gpu_data_code(data_size_dict, data_type, file_name="default_kernel.cu", output_names=["output"]):
  '''
  data_size_dict: dict of data_name and size, e.g. {input: 1024, weight: 2048}
  data_type: the data type the the data, like float, half
  '''
  code = []
  CPU_malloc_code = []
  GPU_declare_code = []
  GPU_malloc_code = []
  GPU_cp_code = []
  GPU_cpback_code = []
  CPU_free_code = []
  GPU_free_code = []
  for name, size in data_size_dict.items():
    var_size = "{}_size".format(name)
    line = "\tconst int {}_size={};".format(name, size)
    CPU_malloc_code.append(line)
    line = "\t{} *{} = new {}[{}];\n".format(data_type, name, data_type, var_size)
    CPU_malloc_code.append(line)
    line = "\t{} *d_{}=NULL;\n".format(data_type, name)
    GPU_declare_code.append(line)
    line = "\terr=cudaMalloc((void **)&d_{}, sizeof({})*{});\n".format(name, data_type, var_size)
    GPU_malloc_code.append(line)
    if name not in output_names:
      line = "\tcudaMemcpy(d_{}, {}, sizeof({})*{}, cudaMemcpyHostToDevice);\n".format(name, name, data_type, var_size)
      GPU_cp_code.append(line)
    if name in output_names:
      line = "\tcudaMemcpy({}, d_{}, sizeof({})*{}, cudaMemcpyDeviceToHost);\n".format(name, name, data_type, var_size)
      GPU_cpback_code.append(line)
    CPU_free_code.append("\tdelete[] {};\n".format(name))
    GPU_free_code.append("\tcudaFree(d_{});\n".format(name))
  
  
  code.append("int main(){\n")
  code.extend(CPU_malloc_code)
  code.append("\n")
  code.append("\tcudaError_t err = cudaSuccess;\n")
  code.extend(GPU_declare_code)
  code.extend(GPU_malloc_code)
  code.append("\n")
  code.extend(GPU_cp_code)
  code.append("\n")
  code.extend("\tcudaDeviceSynchronize();\n")
  code.extend(GPU_cpback_code)
  code.extend("\tcudaDeviceSynchronize();\n")
  code.append("""
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }\n
  """)
  code.extend(CPU_free_code)
  code.extend(GPU_free_code)
  code.append("\treturn 0;\n}")

  f = open(file_name, 'w')
  f.writelines(code)
  f.flush()
  f.close()


def test_generate():
  # generate_gpu_data_code({"input": 1*12*386*64, "weight": 1*12*386*64, "output": 1*12*384*384, \
  #   "intermedia_output": 1*12*384*384, "ori_output": 1*12*384*384}, "half", output_names=["output", "intermedia_output", "ori_output"])
  # batch, num_heads, seq_length, dim = 64, 4, 49, 32
  # generate_gpu_data_code({"input": batch*num_heads*seq_length*dim, "weight": batch*num_heads*seq_length*dim, "output": batch*num_heads*seq_length*seq_length, \
  #   "intermedia_output": batch*num_heads*seq_length*seq_length, "ori_output": batch*num_heads*seq_length*seq_length}, "half", output_names=["output", "intermedia_output", "ori_output"])
  # For cudnn softmax
  # batch, num_heads, seq_length, seq_length = 64, 4, 49, 49
  # generate_gpu_data_code({"input": batch*num_heads*seq_length*seq_length, "output": batch*num_heads*seq_length*seq_length, }, 
  #   "half", output_names=["output"])
  # For swin-transformer query-key matmul
  # batch, num_heads, seq_length, dim = 64, 4, 49, 32
  # generate_gpu_data_code({"input": batch*num_heads*seq_length*dim, "weight": batch*num_heads*seq_length*dim, "output": batch*num_heads*seq_length*seq_length}, "half", output_names=["output"])
  # Fork qkv matmul
  # batch, height, width, channel = 64, 4, 49, 32
  # generate_gpu_data_code({"input": batch*height*width*channel, "weight": 3*channel*channel, "output": batch*width*channel*3*channel}, "half", output_names=["output"])

  # batch, height, width, channel = 1, 16, 16, 512
  # generate_gpu_data_code({"input": batch*height*width*4*channel, "weight": 4*channel*channel, "output": batch*width*channel}, "half", output_names=["output"])

  # M, N, K = 16, 16, 64
  # generate_gpu_data_code({"input":M*K, "weight": N*K, "output": M*N}, "float", output_names=["output"])
  # 16 * 256 * 4 = 16K * 108 * 4
  # chunk, hidden_size = 65536, 256
  # generate_gpu_data_code({"input": chunk*hidden_size, "weight1": hidden_size * hidden_size, 
  #   "weight2": hidden_size * hidden_size, "weight3": hidden_size * hidden_size, 
  #   "weight4": hidden_size * hidden_size, "output": chunk*hidden_size}, "half", output_names=["output"])
  num_head, seq_length, hidden_size_per_head = 12, 128, 64
  # generate_gpu_data_code({"qeury": num_head * seq_length * hidden_size_per_head, "key": num_head * seq_length * hidden_size_per_head, 
  #   "output": num_head * seq_length * seq_length, "sum": num_head * seq_length}, "half", output_names=["output", "sum"])
  
  # generate_gpu_data_code({"input": seq_length*num_head*hidden_size_per_head, "weight_qkv": num_head*hidden_size_per_head*3*num_head*hidden_size_per_head, 
  #   "query": num_head * seq_length * hidden_size_per_head, "key": num_head * seq_length * hidden_size_per_head, 
  #   "value": num_head * seq_length * hidden_size_per_head, 
  #   "output": num_head * seq_length * hidden_size_per_head}, "half", output_names=["output", "query", "key", "value"])
  
  generate_gpu_data_code({
      "input": seq_length*num_head*hidden_size_per_head, 
      "weight_qkv": num_head*hidden_size_per_head*3*num_head*hidden_size_per_head, 
      "qkv_output": 3*num_head*num_head*hidden_size_per_head, 
      "query": num_head * seq_length * hidden_size_per_head, 
      "key": num_head * seq_length * hidden_size_per_head, 
      "value": num_head * seq_length * hidden_size_per_head, 
      "output": num_head * seq_length * hidden_size_per_head,
      "query_key_output": num_head * seq_length * seq_length,
      "sum": num_head * seq_length,
      "qv_value_output": num_head * seq_length * hidden_size_per_head,
      "attn_fc_weight" : num_head * hidden_size_per_head * num_head * hidden_size_per_head,
      "attn_fc_output" : num_head * seq_length * hidden_size_per_head,
      "attn_output_layer_norm_sum" : seq_length,
      "attn_output_layer_norm_variance" : seq_length,
    }, 
    "half", output_names=[
      "output", "query", "key", "value", "query_key_output", "sum", 
      "qv_value_output", "attn_fc_output", "attn_output_layer_norm_sum", "attn_output_layer_norm_variance"])

if __name__=="__main__":
  test_generate()
