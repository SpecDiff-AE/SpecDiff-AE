#include <torch/extension.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

namespace diffspec::amd {
namespace {

template <typename scalar_t>
__global__ void compact_kv_spans_kernel(const scalar_t* __restrict__ src,
                                        scalar_t* __restrict__ dst,
                                        const std::int64_t* __restrict__ spans,
                                        std::int64_t num_spans,
                                        std::int64_t batch,
                                        std::int64_t heads,
                                        std::int64_t src_tokens,
                                        std::int64_t head_dim,
                                        std::int64_t active_len,
                                        std::int64_t dst_stride_b,
                                        std::int64_t dst_stride_h,
                                        std::int64_t dst_stride_t,
                                        std::int64_t dst_stride_d,
                                        std::int64_t total_elems) {
  for (std::int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total_elems;
       linear += std::int64_t(blockDim.x) * gridDim.x) {
    std::int64_t d = linear % head_dim;
    std::int64_t tmp = linear / head_dim;
    std::int64_t out_t = tmp % active_len;
    tmp /= active_len;
    std::int64_t h = tmp % heads;
    std::int64_t b = tmp / heads;

    std::int64_t cursor = 0;
    std::int64_t src_t = -1;
    for (std::int64_t i = 0; i < num_spans; ++i) {
      std::int64_t start = spans[i * 2 + 0];
      std::int64_t end = spans[i * 2 + 1];
      if (start < 0) {
        start = 0;
      }
      if (end > src_tokens) {
        end = src_tokens;
      }
      if (end < start) {
        end = start;
      }
      std::int64_t len = end - start;
      if (out_t < cursor + len) {
        src_t = start + (out_t - cursor);
        break;
      }
      cursor += len;
    }

    if (src_t >= 0) {
      std::int64_t src_idx =
          ((b * heads + h) * src_tokens + src_t) * head_dim + d;
      std::int64_t dst_idx = b * dst_stride_b + h * dst_stride_h +
                             out_t * dst_stride_t + d * dst_stride_d;
      dst[dst_idx] = src[src_idx];
    }
  }
}

}  // namespace

void compact_kv_spans_launcher(torch::Tensor src,
                               torch::Tensor dst,
                               torch::Tensor spans,
                               std::int64_t active_len) {
  if (active_len == 0) {
    return;
  }

  const std::int64_t batch = src.size(0);
  const std::int64_t heads = src.size(1);
  const std::int64_t src_tokens = src.size(2);
  const std::int64_t head_dim = src.size(3);
  const std::int64_t total_elems = batch * heads * active_len * head_dim;
  const std::int64_t dst_stride_b = dst.stride(0);
  const std::int64_t dst_stride_h = dst.stride(1);
  const std::int64_t dst_stride_t = dst.stride(2);
  const std::int64_t dst_stride_d = dst.stride(3);
  const int threads = 256;
  const int blocks = static_cast<int>(
      std::min<std::int64_t>((total_elems + threads - 1) / threads, 65535));
  const c10::cuda::CUDAGuard device_guard(src.device());
  auto stream = at::cuda::getCurrentCUDAStream(src.device().index()).stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "compact_kv_spans", [&] {
    compact_kv_spans_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        src.data_ptr<scalar_t>(),
        dst.data_ptr<scalar_t>(),
        spans.data_ptr<std::int64_t>(),
        spans.size(0),
        batch,
        heads,
        src_tokens,
        head_dim,
        active_len,
        dst_stride_b,
        dst_stride_h,
        dst_stride_t,
        dst_stride_d,
        total_elems);
  });

  C10_HIP_KERNEL_LAUNCH_CHECK();
}

}  // namespace diffspec::amd
