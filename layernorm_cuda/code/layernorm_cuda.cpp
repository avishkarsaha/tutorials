#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA functions
torch::Tensor layer_norm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    return layer_norm_cuda_forward(input, weight, bias, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layer_norm_forward, "LayerNorm forward (CUDA)");
}
