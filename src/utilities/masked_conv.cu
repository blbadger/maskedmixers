#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace extension_cpp {

__global__
void maskedconv_kernel(int numel, int token_index, const float* input, const float* conv_weights, float *output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numel) {
        float result = 0.0f;
        for (int i = 0; i < ; ++i) {
            result += input[i] * conv_weights[i];
            }
        }
        output[tid] = result;
    }
}


at::Tensor maskedconv_cuda(const at::Tensor& input, const at::Tensor& conv_weight, at::Tensor& output) {
  TORCH_CHECK(input.sizes() == out.sizes());
  TORCH_CHECK(input.dtype() == at::kFloat);
  TORCH_CHECK(conv_weight.dtype() == at::kFloat);
  TORCH_CHECK(output.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(conv_weight.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
  at::Tensor input_contig = input.contiguous();
  at::Tensor conv_weight_contig = conv_weight.contiguous();
  const float* input_ptr = input_contig.data_ptr<float>();
  const float* conv_weight_ptr = conv_weight_contig.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();
  int numel = input_contig.numel();
  int token_index = conv_weight.numel()
  maskedconv_kernel<<<(numel+255)/256, 256>>>(numel, token_index, token_index, a_ptr, b_ptr, output_ptr);
  return routput
}

// register implementations
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("maskedconv", &maskedconv_cuda);
}
