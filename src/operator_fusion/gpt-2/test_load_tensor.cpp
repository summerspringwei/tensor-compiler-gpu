// torch lib path: /usr/local/lib/miniconda3/envs/cloud-ai-lab/lib/python3.8/site-packages/torch/__init__.py


#include <torch/torch.h>

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
};


int main(int argc, char* argv[])
{
    std::vector<char> f = get_the_bytes(std::string(argv[1]));
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor tensor = x.toTensor();
    // std::cout << tensor;
    torch::print(tensor);
}
