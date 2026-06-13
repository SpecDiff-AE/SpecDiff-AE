#include <torch/extension.h>

#if defined(DIFFSPEC_WITH_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include <cstdint>
#include <algorithm>
#include <string>

namespace py = pybind11;

namespace diffspec::amd {
void compact_kv_spans_launcher(torch::Tensor src,
                               torch::Tensor dst,
                               torch::Tensor spans,
                               std::int64_t active_len);
}  // namespace diffspec::amd

namespace {

py::dict make_result(bool applied,
                     std::size_t window_bytes,
                     const std::string& reason) {
  py::dict out;
  out["applied"] = applied;
  out["window_bytes"] = static_cast<unsigned long long>(window_bytes);
  out["reason"] = reason;
  return out;
}

void clear_last_hip_error() {
#if defined(DIFFSPEC_WITH_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  (void)hipGetLastError();
#endif
}

int device_attribute_or_zero(hipDeviceAttribute_t attr, int device_index) {
#if defined(DIFFSPEC_WITH_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (device_index < 0) {
    if (hipGetDevice(&device_index) != hipSuccess) {
      clear_last_hip_error();
      return 0;
    }
  }
  int value = 0;
  hipError_t err = hipDeviceGetAttribute(&value, attr, device_index);
  if (err != hipSuccess) {
    clear_last_hip_error();
    return 0;
  }
  return value;
#else
  (void)attr;
  (void)device_index;
  return 0;
#endif
}

void validate_tensor_pair(const torch::Tensor& src,
                          const torch::Tensor& dst,
                          const torch::Tensor& spans,
                          std::int64_t active_len) {
  TORCH_CHECK(src.is_cuda(), "src must be a ROCm/CUDA tensor");
  TORCH_CHECK(dst.is_cuda(), "dst must be a ROCm/CUDA tensor");
  TORCH_CHECK(spans.is_cuda(), "spans must be a ROCm/CUDA tensor");
  TORCH_CHECK(src.scalar_type() == dst.scalar_type(),
              "src and dst dtype must match");
  TORCH_CHECK(src.dim() == 4, "src must have shape [B, H, T, D]");
  TORCH_CHECK(dst.dim() == 4, "dst must have shape [B, H, active_T, D]");
  TORCH_CHECK(spans.dim() == 2 && spans.size(1) == 2,
              "spans must have shape [N, 2] with start/end offsets");
  TORCH_CHECK(spans.scalar_type() == torch::kInt64,
              "spans must be int64");
  TORCH_CHECK(src.size(0) == dst.size(0), "batch mismatch");
  TORCH_CHECK(src.size(1) == dst.size(1), "head mismatch");
  TORCH_CHECK(src.size(3) == dst.size(3), "head_dim mismatch");
  TORCH_CHECK(active_len >= 0, "active_len must be non-negative");
  TORCH_CHECK(active_len <= dst.size(2), "active_len exceeds dst sequence dim");
  TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
  TORCH_CHECK(spans.is_contiguous(), "spans must be contiguous");
}

}  // namespace

py::dict set_access_policy_window(std::uint64_t stream_handle,
                                  std::uint64_t base_ptr,
                                  std::size_t window_bytes,
                                  double hit_ratio,
                                  int device_index) {
#if defined(DIFFSPEC_WITH_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (base_ptr == 0 || window_bytes == 0) {
    return make_result(false, 0, "empty_window");
  }

  const int access_policy_max_window_size =
      device_attribute_or_zero(hipDeviceAttributeAccessPolicyMaxWindowSize, device_index);
  const int persisting_l2_cache_max_size =
      device_attribute_or_zero(hipDeviceAttributePersistingL2CacheMaxSize, device_index);
  const int l2_cache_size =
      device_attribute_or_zero(hipDeviceAttributeL2CacheSize, device_index);
  if (access_policy_max_window_size <= 0) {
    py::dict out = make_result(false, 0, "access_policy_window_unsupported");
    out["access_policy_max_window_size"] = access_policy_max_window_size;
    out["persisting_l2_cache_max_size"] = persisting_l2_cache_max_size;
    out["l2_cache_size"] = l2_cache_size;
    return out;
  }

  window_bytes = std::min<std::size_t>(
      window_bytes, static_cast<std::size_t>(access_policy_max_window_size));

  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_handle);
  hipStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(base_ptr);
  attr.accessPolicyWindow.num_bytes = window_bytes;
  attr.accessPolicyWindow.hitRatio = static_cast<float>(hit_ratio);
  attr.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = hipAccessPropertyStreaming;

  hipError_t err = hipStreamSetAttribute(
      stream, hipStreamAttributeAccessPolicyWindow, &attr);
  if (err != hipSuccess) {
    std::string reason = hipGetErrorString(err);
    clear_last_hip_error();
    py::dict out = make_result(false, 0, reason);
    out["access_policy_max_window_size"] = access_policy_max_window_size;
    out["persisting_l2_cache_max_size"] = persisting_l2_cache_max_size;
    out["l2_cache_size"] = l2_cache_size;
    return out;
  }
  py::dict out = make_result(true, window_bytes, "");
  out["access_policy_max_window_size"] = access_policy_max_window_size;
  out["persisting_l2_cache_max_size"] = persisting_l2_cache_max_size;
  out["l2_cache_size"] = l2_cache_size;
  return out;
#else
  (void)stream_handle;
  (void)base_ptr;
  (void)window_bytes;
  (void)hit_ratio;
  (void)device_index;
  return make_result(false, 0, "not_built_with_hip");
#endif
}

py::dict clear_access_policy_window(std::uint64_t stream_handle) {
#if defined(DIFFSPEC_WITH_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  hipStream_t stream = reinterpret_cast<hipStream_t>(stream_handle);
  hipStreamAttrValue attr{};
  attr.accessPolicyWindow.num_bytes = 0;
  hipError_t err = hipStreamSetAttribute(
      stream, hipStreamAttributeAccessPolicyWindow, &attr);
  if (err != hipSuccess) {
    std::string reason = hipGetErrorString(err);
    clear_last_hip_error();
    return make_result(false, 0, reason);
  }
  return make_result(true, 0, "");
#else
  (void)stream_handle;
  return make_result(false, 0, "not_built_with_hip");
#endif
}

py::dict compact_kv_spans(torch::Tensor src,
                          torch::Tensor dst,
                          torch::Tensor spans,
                          std::int64_t active_len) {
  validate_tensor_pair(src, dst, spans, active_len);
#if defined(DIFFSPEC_WITH_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  diffspec::amd::compact_kv_spans_launcher(src, dst, spans, active_len);
  py::dict out;
  out["applied"] = true;
  out["backend"] = "hip_compact_kv_spans";
  out["active_len"] = active_len;
  out["spans"] = spans.size(0);
  return out;
#else
  (void)src;
  (void)dst;
  (void)spans;
  (void)active_len;
  py::dict out;
  out["applied"] = false;
  out["backend"] = "none";
  out["reason"] = "not_built_with_hip";
  return out;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_access_policy_window", &set_access_policy_window);
  m.def("clear_access_policy_window", &clear_access_policy_window);
  m.def("compact_kv_spans", &compact_kv_spans);
}
