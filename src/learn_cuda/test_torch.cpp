#include <iostream>
#include <sstream>
#include <vector>

#include "torch/all.h"

int main(int argc, char** argv) {
  torch::Tensor tensor = torch::rand({2, 3});
  auto out = torch::mean(tensor, {1, });

  torch::print(tensor);
  torch::print(out);
  return 0;
}