#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <assert.h>

#include "npy.hpp"

float sigmoid_f(float x) { return (1.0f / (1 + exp(-x))); }

template <int num_hidden>
void gemv(std::shared_ptr<std::vector<float>> input,
          std::shared_ptr<std::vector<float>> weight,
          std::shared_ptr<std::vector<float>> output) {
  for (int i = 0; i < num_hidden; ++i) {
    output->at(i) = 0;
    for (int j = 0; j < num_hidden; ++j) {
      output->at(i) += input->at(j) * weight->at(i * num_hidden + j);
    }
  }
}

std::shared_ptr<std::vector<float>>
vadd(std::shared_ptr<std::vector<float>> va,
     std::shared_ptr<std::vector<float>> vb) {
  assert(va->size() == vb->size());
  std::shared_ptr<std::vector<float>> vc =
      std::make_shared<std::vector<float>>(va->size());
  for (int i = 0; i < va->size(); ++i) {
    vc->at(i) = va->at(i) + vb->at(i);
  }
  return vc;
}

std::shared_ptr<std::vector<float>> vadd(std::shared_ptr<std::vector<float>> va,
                                         float num) {
  std::shared_ptr<std::vector<float>> vc =
      std::make_shared<std::vector<float>>(va->size());
  for (int i = 0; i < va->size(); ++i) {
    vc->at(i) = va->at(i) + num;
  }
  return vc;
}

std::shared_ptr<std::vector<float>>
vmul(std::shared_ptr<std::vector<float>> va,
     std::shared_ptr<std::vector<float>> vb) {
  assert(va->size() == vb->size());
  std::shared_ptr<std::vector<float>> vc =
      std::make_shared<std::vector<float>>(va->size());
  for (int i = 0; i < va->size(); ++i) {
    vc->at(i) = va->at(i) * vb->at(i);
  }
  return vc;
}

enum Activation {
  Activation_sigmoid,
  Activation_tanh,
};

std::shared_ptr<std::vector<float>>
activate_in_place(std::shared_ptr<std::vector<float>> va, Activation type) {
  switch (type) {
  case Activation::Activation_sigmoid:
    for (int i = 0; i < va->size(); ++i) {
      va->at(i) = sigmoid_f(va->at(i));
    }
    break;
  case Activation::Activation_tanh:
    for (int i = 0; i < va->size(); ++i) {
      va->at(i) = std::tanh(va->at(i));
    }
    break;
  default:
    printf("activation type does not support\n");
    break;
  }
  return va;
}

std::shared_ptr<std::vector<float>>
activate(std::shared_ptr<std::vector<float>> va, Activation type) {
  std::shared_ptr<std::vector<float>> vc =
      std::make_shared<std::vector<float>>(va->size());
  switch (type) {
  case Activation::Activation_sigmoid:
    for (int i = 0; i < va->size(); ++i) {
      vc->at(i) = sigmoid_f(va->at(i));
    }
    break;
  case Activation::Activation_tanh:
    for (int i = 0; i < va->size(); ++i) {
      vc->at(i) = std::tanh(va->at(i));
    }
    break;
  default:
    printf("activation type does not support\n");
    break;
  }
  return vc;
}

template <int num_hidden>
std::pair<std::shared_ptr<std::vector<float>>,
          std::shared_ptr<std::vector<float>>>
lstm_cell(std::shared_ptr<std::vector<float>> inputs,
          std::shared_ptr<std::vector<float>> c_state,
          std::shared_ptr<std::vector<float>> h_state,
          std::shared_ptr<std::vector<std::shared_ptr<std::vector<float>>>> W,
          std::shared_ptr<std::vector<std::shared_ptr<std::vector<float>>>> U,
          std::shared_ptr<std::vector<float>> bias) {
  const int kNumInputGates = 4;
  std::shared_ptr<std::vector<std::shared_ptr<std::vector<float>>>>
      input_gate_outputs =
          std::make_shared<std::vector<std::shared_ptr<std::vector<float>>>>(
              kNumInputGates);
  std::shared_ptr<std::vector<std::shared_ptr<std::vector<float>>>>
      state_gate_outputs =
          std::make_shared<std::vector<std::shared_ptr<std::vector<float>>>>(
              kNumInputGates);
  std::shared_ptr<std::vector<std::shared_ptr<std::vector<float>>>> output_buf =
      std::make_shared<std::vector<std::shared_ptr<std::vector<float>>>>(
          kNumInputGates);
  for (int i = 0; i < kNumInputGates; ++i) {
    input_gate_outputs->at(i) = (
        std::make_shared<std::vector<float>>(num_hidden));
    state_gate_outputs->at(i) = (
        std::make_shared<std::vector<float>>(num_hidden));
  }
  // TODO(xiachunwei) Set value
  for (int i = 0; i < kNumInputGates; ++i) {
    gemv<num_hidden>(inputs, W->at(i), input_gate_outputs->at(i));
    gemv<num_hidden>(h_state, U->at(i), state_gate_outputs->at(i));
  }

  auto i =
      vadd(vadd(input_gate_outputs->at(0), state_gate_outputs->at(0)), bias);
  auto j =
      vadd(vadd(input_gate_outputs->at(1), state_gate_outputs->at(1)), bias);
  auto f =
      vadd(vadd(input_gate_outputs->at(2), state_gate_outputs->at(2)), bias);
  auto o =
      vadd(vadd(input_gate_outputs->at(3), state_gate_outputs->at(3)), bias);

  auto new_c =
      vadd(vmul(c_state, activate(f, Activation::Activation_sigmoid)),
           vmul(activate(i, Activation::Activation_sigmoid),
                activate(j, Activation::Activation_tanh)));
  auto new_h = vmul(activate(new_c, Activation::Activation_tanh),
                    activate(o, Activation::Activation_tanh));

  return std::make_pair(new_c, new_h);
}

// template<int batch, int num_layer, int num_hidden, int num_timestep>
// void lstm_cpu(float* inputs_timestep, float* outputs_timestep,
//     float* c_wavefront, float* h_wavefront, float* input_wavefront,
//     float* weight_input_wavefront, float* weight_state_wavefront, float*
//     bias, float* output_buffer){

// };

void print_vector(std::shared_ptr<std::vector<float>> vec) {
  for (int i = 0; i < vec->size(); ++i) {
    printf("%f ", vec->at(i));
  }
  printf("\n");
}

void test_gemv() {
  const int num_hidden = 4;
  auto input = std::make_shared<std::vector<float>>(num_hidden);
  auto weight = std::make_shared<std::vector<float>>(num_hidden * num_hidden);
  auto output = std::make_shared<std::vector<float>>(num_hidden);
  for (int i = 0; i < num_hidden; ++i) {
    input->at(i) = 1;
    for (int j = 0; j < num_hidden; ++j) {
      weight->at(i * num_hidden + j) = 1;
    }
  }
  gemv<num_hidden>(input, weight, output);
  print_vector(output);
  auto output_add = vadd(input, input);
  print_vector(output_add);
  output_add = vadd(input, 2);
  print_vector(output_add);
  auto sigmoid_output =
      activate_in_place(output_add, Activation::Activation_sigmoid);
  print_vector(output_add);
  output_add = activate(output_add, Activation_sigmoid);
  print_vector(output_add);
}

void test_lstm_cell(){
    using sp_vector = std::shared_ptr<std::vector<float>>;
    unsigned long num_hidden=256;
    const size_t kNumGate = 4;
    // Prepare data
    auto input = std::make_shared<std::vector<float>>(num_hidden);
    auto c_state = std::make_shared<std::vector<float>>(num_hidden);
    auto h_state = std::make_shared<std::vector<float>>(num_hidden);
    auto bias = std::make_shared<std::vector<float>>(num_hidden);
    auto W = std::make_shared<std::vector<sp_vector>>(kNumGate);
    auto U = std::make_shared<std::vector<sp_vector>>(kNumGate);
    for(int i=0; i<kNumGate; ++i){
        W->at(i) = (std::make_shared<std::vector<float>>(num_hidden*num_hidden));
        U->at(i) = (std::make_shared<std::vector<float>>(num_hidden*num_hidden));
    }
    // Load from npy file
    std::vector<unsigned long> input_shape = {num_hidden};
    std::vector<unsigned long> weight_shape = {num_hidden * num_hidden};
    
    bool fortran_order;
    npy::LoadArrayFromNumpy(std::string("data/input.npy"), input_shape, fortran_order, *(input.get()));
    npy::LoadArrayFromNumpy(std::string("data/c_state.npy"), input_shape, fortran_order, *c_state.get());
    npy::LoadArrayFromNumpy(std::string("data/h_state.npy"), input_shape, fortran_order, *h_state.get());
    npy::LoadArrayFromNumpy(std::string("data/bias.npy"), input_shape, fortran_order, *bias.get());
    for(int i=0; i<kNumGate; ++i){
        char buf[128];
        snprintf(buf, sizeof(buf), "data/W_%d.npy", i);
        npy::LoadArrayFromNumpy<float>(std::string(buf), weight_shape, fortran_order, *(W->at(i).get()));
        // print_vector(W->at(i));

        snprintf(buf, sizeof(buf), "data/U_%d.npy", i);
        npy::LoadArrayFromNumpy<float>(std::string(buf), weight_shape, fortran_order, *(U->at(i).get()));
        // print_vector(U->at(i));
    }
    printf("Load data done!\n");
    // First layer
    auto cell_output = lstm_cell<256>(input, c_state, h_state, W, U, bias);
    printf("c_state:\n");
    print_vector(cell_output.first);
    printf("h_state:\n");
    print_vector(cell_output.second);
    for(int i=0; i<9;++i){
        cell_output = lstm_cell<256>(cell_output.second, cell_output.first, cell_output.second, W, U, bias);
    }
    printf("c_state:\n");
    print_vector(cell_output.first);
    printf("h_state:\n");
    print_vector(cell_output.second);
}


void test_lstm_timesteps(){
    using sp_vector = std::shared_ptr<std::vector<float>>;
    unsigned long num_hidden=256;
    const size_t kNumGate = 4;
    const size_t num_timesteps=100;
    const size_t num_layer = 10;
    // Prepare data
    auto input = std::make_shared<std::vector<float>>(num_hidden);
    auto c_state = std::make_shared<std::vector<float>>(num_hidden);
    auto h_state = std::make_shared<std::vector<float>>(num_hidden);
    auto bias = std::make_shared<std::vector<float>>(num_hidden);
    auto W = std::make_shared<std::vector<sp_vector>>(kNumGate);
    auto U = std::make_shared<std::vector<sp_vector>>(kNumGate);
    for(int i=0; i<kNumGate; ++i){
        W->at(i) = (std::make_shared<std::vector<float>>(num_hidden*num_hidden));
        U->at(i) = (std::make_shared<std::vector<float>>(num_hidden*num_hidden));
    }
    // Load from npy file
    std::vector<unsigned long> input_shape = {num_hidden};
    std::vector<unsigned long> weight_shape = {num_hidden * num_hidden};
    
    bool fortran_order;
    npy::LoadArrayFromNumpy(std::string("data/input.npy"), input_shape, fortran_order, *(input.get()));
    npy::LoadArrayFromNumpy(std::string("data/c_state.npy"), input_shape, fortran_order, *c_state.get());
    npy::LoadArrayFromNumpy(std::string("data/h_state.npy"), input_shape, fortran_order, *h_state.get());
    npy::LoadArrayFromNumpy(std::string("data/bias.npy"), input_shape, fortran_order, *bias.get());
    for(int i=0; i<kNumGate; ++i){
        char buf[128];
        snprintf(buf, sizeof(buf), "data/W_%d.npy", i);
        npy::LoadArrayFromNumpy<float>(std::string(buf), weight_shape, fortran_order, *(W->at(i).get()));
        snprintf(buf, sizeof(buf), "data/U_%d.npy", i);
        npy::LoadArrayFromNumpy<float>(std::string(buf), weight_shape, fortran_order, *(U->at(i).get()));
    }
    printf("Load data done!\n");
    std::vector<sp_vector> input_timesteps(num_timesteps);
    for(int i=0; i<num_timesteps; ++i){
        input_timesteps.at(i) = input;
    }
    std::vector<std::pair<sp_vector, sp_vector>> lstm_layers_states(num_layer);
    for(int i=0; i<num_layer; ++i){
        lstm_layers_states[i] = std::make_pair(c_state, h_state);
    }
    // First layer
    for(int step=0; step<num_timesteps; ++step){
        lstm_layers_states[0] = lstm_cell<256>(input, lstm_layers_states[0].first, lstm_layers_states[0].second, W, U, bias);
        for(int i=1; i<9;++i){
            lstm_layers_states[i] = lstm_cell<256>(lstm_layers_states[0].second, lstm_layers_states[1].first, lstm_layers_states[1].second, W, U, bias);
        }
    }
    
}

int main() {
//   test_gemv();
  test_lstm_cell();
  return 0;
}
