

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
    line = "\t{} *{} = new {}[{}];\n".format(data_type, name, data_type, size)
    CPU_malloc_code.append(line)
    line = "\t{} *{}=NULL;\n".format(data_type, name)
    GPU_declare_code.append(line)
    line = "\terr=cudaMalloc((void **)&d_{}, sizeof({})*{});\n".format(name, data_type, size)
    GPU_malloc_code.append(line)
    if name not in output_names:
      line = "\tcudaMemcpy(d_{}, {}, sizeof({})*{}, cudaMemcpyHostToDevice);\n".format(name, name, data_type, size)
      GPU_cp_code.append(line)
    if name in output_names:
      line = "\tcudaMemcpy(d_{}, {}, sizeof({})*{}, cudaMemcpyDeviceToHost);\n".format(name, name, data_type, size)
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
  code.extend(CPU_free_code)
  code.extend(GPU_free_code)
  code.append("\treturn 0;\n}")

  f = open(file_name, 'w')
  f.writelines(code)
  f.flush()
  f.close()


def test_generate():
  generate_gpu_data_code({"input": 1024, "weight": 2048, "output": 1024}, "float")


if __name__=="__main__":
  test_generate()
