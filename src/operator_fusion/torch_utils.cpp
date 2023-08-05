#include "torch_utils.h"

#include <iostream>
#include <sstream>


std::string idx2cordinate(uint64_t idx, std::vector<uint64_t>& acc_mul){
  std::vector<uint64_t> coordinate;
  const int dim = acc_mul.size();
  for(int i=0; i<dim-1; ++i){
    coordinate.push_back(idx / acc_mul[i+1]);
    idx = idx % acc_mul[i+1];
  }
  coordinate.push_back(idx);
  std::stringstream ss;
  // printf("(");
  ss << "(";
  for(int j=0; j<coordinate.size(); ++j){
    // printf("%u ", coordinate[j]);
    ss<<coordinate[j]<<" ";
  }
  ss <<")";
  // printf(")");
  return ss.str();
}


void my_compare(torch::Tensor& a, torch::Tensor& b, float rotl, float aotl, int print_detail){
  auto shape = a.sizes();
  const int dim = shape.size();
  std::vector<uint64_t> acc_mul(dim);
  acc_mul[dim-1] = shape[dim-1]; // acc_mul[2] = shape[2]
  for(int i=0; i<dim-1; ++i){
    acc_mul[dim-2-i] = acc_mul[dim-i-1] * shape[dim-2-i]; 
  }
  char* output_buff=(char*)malloc(1024*1024*128);
  char f_buff[32];
  size_t offset = 0;
  // std::stringstream ss;
  int error_cnt = 0;
  auto num_elements = a.numel();
  auto reshaped_a = torch::reshape(a, {num_elements, });
  auto reshaped_b = torch::reshape(b, {num_elements, });
  for(uint64_t i=0; i<num_elements; ++i){
    // auto x = reshaped_a[i].item().toHalf();
    // auto y = reshaped_b[i].item().toHalf();
    auto x = reshaped_a[i].item().toFloat();
    auto y = reshaped_b[i].item().toFloat();
    auto left = std::abs(x - y);
    auto right = rotl * std::abs(x) + aotl;
    printf("%f < %f ? %d\n", left, right, left < right);
    if(left > right){
      error_cnt ++;
      if(print_detail>=1){
        auto str_coord = idx2cordinate(i, acc_mul);
        sprintf(output_buff+offset, "%s %s", "diff ", str_coord.c_str());offset+=(5+str_coord.length());
        sprintf(f_buff, "%f %f\n", x, y);
        sprintf(output_buff+offset, "%s", f_buff);offset+=(strlen(f_buff));
      }
    }else{
      if(print_detail>=2){
        auto str_coord = idx2cordinate(i, acc_mul);
        sprintf(output_buff+offset, "%s %s", "same ", str_coord.c_str());offset+=(5+str_coord.length());
        sprintf(f_buff, "%f %f\n", x, y);
        sprintf(output_buff+offset, "%s", f_buff);offset+=(strlen(f_buff));
        // ss << "same " << idx2cordinate(i, acc_mul);
        // ss << x << " " << y << "\n";
        // ss << __half2float(x) << " " << __half2float(y) << "\n";
        // printf("same ");
        // idx2cordinate(i, acc_mul);
        // printf(" %f %f\n", __half2float(x), __half2float(y));
      }
    }
  }
  // printf("%s\n", ss.str().c_str());
  printf("%s\n", output_buff);
  printf("my_compare error_cnt %d, total %d, error ratio %.3f\n", error_cnt, num_elements, ((float)error_cnt) / ((float)num_elements));
}

std::vector<char> readFile(const char* filename)
{
    // open the file:
    std::ifstream file(filename, std::ios::binary);

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // reserve capacity
    std::vector<char> vec;
    vec.reserve(fileSize);

    // read the data:
    vec.insert(vec.begin(),
               std::istream_iterator<char>(file),
               std::istream_iterator<char>());

    return vec;
}

torch::Tensor torch_load_tensor(std::string filename){
  
  std::ifstream input(filename, std::ios::binary);
  if(!input.good()){
    printf("In torch_utils.cpp, torch_load_tensor, file %s not exist\n", filename.c_str());
    exit(0);
  }
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));
  input.close();

  torch::IValue x = torch::pickle_load(bytes);
  torch::Tensor tensor = x.toTensor();
  std::stringstream ss;
  ss << "In torch_utils.cpp:91, tensor.dtype() " << tensor.dtype() << " tensor.sizes(): ";
  for(auto s: tensor.sizes()){
    ss << s << " ";
  }ss << "\n";
  return tensor;
}


// torch::Tensor torch_load_model(std::string filename){
//     // torch::jit::script::Module module;
//   try {
//     // Deserialize the ScriptModule from a file using torch::jit::load().
//     auto module = torch::jit::load(filename);
//     for (auto p :
//        module.named_parameters(/*recurse=*/true)) {
//     std::cout << p.name << " shape: ";
//     for(auto s: p.value.sizes()){
//       std::cout << s << " ";
//     }std::cout<<std::endl;
//   }
//   }
//   catch (const c10::Error& e) {
//     std::cerr << "error loading the model\n";
//   }
//   return torch::ones({1,1}, options_fp16);;
// }