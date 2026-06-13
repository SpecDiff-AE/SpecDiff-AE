#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << hipGetErrorString(_err) << std::endl;                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

struct Args {
  std::string mode = "baseline_sparse";
  int full_tokens = 131072;
  int active_chunks = 128;
  int chunk_size = 64;
  int heads = 8;
  int head_dim = 64;
  int kv_parts = 2;
  int passes = 4;
  int warmup = 8;
  int timing_iters = 30;
  int block_threads = 256;
  int lds_tile_bytes = 16384;
  int lds_vector_width_bytes = 16;
  float apw_hit_ratio = 0.85f;
  unsigned int seed = 20260612u;
};

bool get_arg(int argc, char** argv, const char* name, std::string* value) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      *value = argv[i + 1];
      return true;
    }
  }
  return false;
}

template <typename T>
void read_arg(int argc, char** argv, const char* name, T* out) {
  std::string value;
  if (!get_arg(argc, argv, name, &value)) {
    return;
  }
  if constexpr (std::is_same<T, std::string>::value) {
    *out = value;
  } else if constexpr (std::is_same<T, float>::value) {
    *out = std::stof(value);
  } else if constexpr (std::is_same<T, unsigned int>::value) {
    *out = static_cast<unsigned int>(std::stoul(value));
  } else {
    *out = static_cast<T>(std::stoll(value));
  }
}

Args parse_args(int argc, char** argv) {
  Args args;
  read_arg(argc, argv, "--mode", &args.mode);
  read_arg(argc, argv, "--full-tokens", &args.full_tokens);
  read_arg(argc, argv, "--active-chunks", &args.active_chunks);
  read_arg(argc, argv, "--chunk-size", &args.chunk_size);
  read_arg(argc, argv, "--heads", &args.heads);
  read_arg(argc, argv, "--head-dim", &args.head_dim);
  read_arg(argc, argv, "--kv-parts", &args.kv_parts);
  read_arg(argc, argv, "--passes", &args.passes);
  read_arg(argc, argv, "--warmup", &args.warmup);
  read_arg(argc, argv, "--timing-iters", &args.timing_iters);
  read_arg(argc, argv, "--block-threads", &args.block_threads);
  read_arg(argc, argv, "--lds-tile-bytes", &args.lds_tile_bytes);
  read_arg(argc, argv, "--apw-hit-ratio", &args.apw_hit_ratio);
  read_arg(argc, argv, "--seed", &args.seed);
  return args;
}

__global__ void init_linear_kernel(float* data, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x) {
    uint32_t x = static_cast<uint32_t>(i * 1664525ull + 1013904223ull);
    data[i] = static_cast<float>(x & 0xffffu) * 1.52587890625e-5f;
  }
}

__global__ void stage_arena_kernel(const float* __restrict__ full,
                                   const int* __restrict__ indices,
                                   float* __restrict__ arena,
                                   int elems_per_token,
                                   size_t active_elems) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < active_elems;
       i += size_t(blockDim.x) * gridDim.x) {
    int token_slot = static_cast<int>(i / elems_per_token);
    int component = static_cast<int>(i - size_t(token_slot) * elems_per_token);
    int token = indices[token_slot];
    arena[i] = full[size_t(token) * elems_per_token + component];
  }
}

__global__ void baseline_sparse_kernel(const float* __restrict__ full,
                                       const int* __restrict__ indices,
                                       float* __restrict__ out,
                                       float* __restrict__ sink,
                                       int active_tokens,
                                       int elems_per_token,
                                       size_t active_elems,
                                       int passes) {
  float local = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < active_elems;
       i += size_t(blockDim.x) * gridDim.x) {
    int token_slot = static_cast<int>(i / elems_per_token);
    int component = static_cast<int>(i - size_t(token_slot) * elems_per_token);
    float acc = 0.0f;
    for (int pass = 0; pass < passes; ++pass) {
      int permuted_slot = (token_slot + pass * 9973) % active_tokens;
      int token = indices[permuted_slot];
      acc += full[size_t(token) * elems_per_token + component];
    }
    out[i] = acc;
    local += acc * 1.0e-7f;
  }
  if (threadIdx.x == 0) {
    atomicAdd(sink, local);
  }
}

__global__ void arena_direct_kernel(const float* __restrict__ arena,
                                    float* __restrict__ out,
                                    float* __restrict__ sink,
                                    size_t active_elems,
                                    int passes) {
  float local = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < active_elems;
       i += size_t(blockDim.x) * gridDim.x) {
    float acc = 0.0f;
    for (int pass = 0; pass < passes; ++pass) {
      acc += arena[(i + size_t(pass) * 4096u) % active_elems];
    }
    out[i] = acc;
    local += acc * 1.0e-7f;
  }
  if (threadIdx.x == 0) {
    atomicAdd(sink, local);
  }
}

__global__ void arena_lds_kernel(const float* __restrict__ arena,
                                 float* __restrict__ out,
                                 float* __restrict__ sink,
                                 size_t active_elems,
                                 int passes,
                                 int tile_elems) {
  extern __shared__ float lds[];
  constexpr int kVectorElems = 4;
  float local = 0.0f;
  size_t tiles = (active_elems + tile_elems - 1) / tile_elems;

  for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    size_t start = tile * size_t(tile_elems);
    size_t remaining = active_elems - start;
    int count = static_cast<int>(
        remaining < size_t(tile_elems) ? remaining : size_t(tile_elems));

    int vec_count = count / kVectorElems;
    auto* lds_vec = reinterpret_cast<float4*>(lds);
    const auto* arena_vec = reinterpret_cast<const float4*>(arena + start);
    for (int vec_idx = threadIdx.x; vec_idx < vec_count; vec_idx += blockDim.x) {
      lds_vec[vec_idx] = arena_vec[vec_idx];
    }
    for (int local_idx = vec_count * kVectorElems + threadIdx.x;
         local_idx < count;
         local_idx += blockDim.x) {
      lds[local_idx] = arena[start + local_idx];
    }
    __syncthreads();

    for (int local_idx = threadIdx.x; local_idx < count; local_idx += blockDim.x) {
      float acc = 0.0f;
      for (int pass = 0; pass < passes; ++pass) {
        int reuse_idx = (local_idx + pass * 257) % count;
        acc += lds[reuse_idx];
      }
      out[start + local_idx] = acc;
      local += acc * 1.0e-7f;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(sink, local);
  }
}

int device_attribute_or_zero(hipDeviceAttribute_t attr) {
  int device = 0;
  hipError_t device_err = hipGetDevice(&device);
  if (device_err != hipSuccess) {
    (void)hipGetLastError();
    return 0;
  }
  int value = 0;
  hipError_t attr_err = hipDeviceGetAttribute(&value, attr, device);
  if (attr_err != hipSuccess) {
    (void)hipGetLastError();
    return 0;
  }
  return value;
}

void configure_apw(hipStream_t stream,
                   const void* ptr,
                   size_t bytes,
                   float hit_ratio,
                   bool* enabled,
                   size_t* window_bytes,
                   int* access_policy_max_window_size,
                   int* persisting_l2_cache_max_size,
                   int* l2_cache_size,
                   std::string* error) {
  *enabled = false;
  *window_bytes = 0;
  *error = "";
  *access_policy_max_window_size =
      device_attribute_or_zero(hipDeviceAttributeAccessPolicyMaxWindowSize);
  *persisting_l2_cache_max_size =
      device_attribute_or_zero(hipDeviceAttributePersistingL2CacheMaxSize);
  *l2_cache_size = device_attribute_or_zero(hipDeviceAttributeL2CacheSize);
  if (*access_policy_max_window_size <= 0) {
    *error = "access_policy_window_unsupported";
    return;
  }
  bytes = std::min(bytes, static_cast<size_t>(*access_policy_max_window_size));
  hipStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
  attr.accessPolicyWindow.num_bytes = bytes;
  attr.accessPolicyWindow.hitRatio = hit_ratio;
  attr.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = hipAccessPropertyStreaming;
  hipError_t err = hipStreamSetAttribute(
      stream, hipStreamAttributeAccessPolicyWindow, &attr);
  if (err == hipSuccess) {
    *enabled = true;
    *window_bytes = bytes;
  } else {
    *error = hipGetErrorString(err);
    (void)hipGetLastError();
  }
}

void launch_target(const Args& args,
                   hipStream_t stream,
                   const float* full,
                   const int* indices,
                   const float* arena,
                   float* out,
                   float* sink,
                   int active_tokens,
                   int elems_per_token,
                   size_t active_elems,
                   int grid_blocks) {
  if (args.mode == "baseline_sparse") {
    baseline_sparse_kernel<<<grid_blocks, args.block_threads, 0, stream>>>(
        full, indices, out, sink, active_tokens, elems_per_token, active_elems, args.passes);
  } else if (args.mode == "arena" || args.mode == "arena_apw") {
    arena_direct_kernel<<<grid_blocks, args.block_threads, 0, stream>>>(
        arena, out, sink, active_elems, args.passes);
  } else if (args.mode == "arena_apw_lds" || args.mode == "arena_apw_tma") {
    int tile_bytes = std::max(16, (args.lds_tile_bytes / 16) * 16);
    int tile_elems = tile_bytes / static_cast<int>(sizeof(float));
    arena_lds_kernel<<<grid_blocks, args.block_threads, tile_bytes, stream>>>(
        arena, out, sink, active_elems, args.passes, tile_elems);
  } else {
    std::cerr << "Unknown mode: " << args.mode << std::endl;
    std::exit(2);
  }
}

std::vector<int> make_sparse_indices(const Args& args, int active_tokens) {
  std::mt19937 rng(args.seed);
  std::vector<int> indices(active_tokens);
  int max_start = std::max(1, args.full_tokens - args.chunk_size);
  std::uniform_int_distribution<int> chunk_dist(0, max_start / args.chunk_size - 1);
  for (int c = 0; c < args.active_chunks; ++c) {
    int chunk_id = chunk_dist(rng);
    int base = chunk_id * args.chunk_size;
    for (int t = 0; t < args.chunk_size; ++t) {
      int slot = c * args.chunk_size + t;
      if (slot < active_tokens) {
        indices[slot] = std::min(args.full_tokens - 1, base + t);
      }
    }
  }
  std::shuffle(indices.begin(), indices.end(), rng);
  return indices;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  if (args.active_chunks <= 0 || args.chunk_size <= 0 || args.full_tokens <= 0 ||
      args.heads <= 0 || args.head_dim <= 0 || args.kv_parts <= 0) {
    std::cerr << "All size arguments must be positive." << std::endl;
    return 2;
  }

  int active_tokens = args.active_chunks * args.chunk_size;
  if (active_tokens > args.full_tokens) {
    std::cerr << "active_tokens must be <= full_tokens." << std::endl;
    return 2;
  }

  int elems_per_token = args.heads * args.head_dim * args.kv_parts;
  size_t full_elems = size_t(args.full_tokens) * elems_per_token;
  size_t active_elems = size_t(active_tokens) * elems_per_token;
  int grid_blocks = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &grid_blocks, arena_direct_kernel, args.block_threads, 0));
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  int shared_mem_per_block = static_cast<int>(props.sharedMemPerBlock);
  args.lds_tile_bytes = std::max(
      16,
      std::min(args.lds_tile_bytes, (shared_mem_per_block / 16) * 16));
  grid_blocks = std::max(1, grid_blocks * props.multiProcessorCount);

  float* full = nullptr;
  float* arena = nullptr;
  float* out = nullptr;
  float* sink = nullptr;
  int* indices = nullptr;
  HIP_CHECK(hipMalloc(&full, full_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&arena, active_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&out, active_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&sink, sizeof(float)));
  HIP_CHECK(hipMalloc(&indices, size_t(active_tokens) * sizeof(int)));
  HIP_CHECK(hipMemset(sink, 0, sizeof(float)));

  auto host_indices = make_sparse_indices(args, active_tokens);
  HIP_CHECK(hipMemcpy(indices, host_indices.data(), size_t(active_tokens) * sizeof(int),
                      hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  init_linear_kernel<<<grid_blocks, args.block_threads, 0, stream>>>(full, full_elems);
  stage_arena_kernel<<<grid_blocks, args.block_threads, 0, stream>>>(
      full, indices, arena, elems_per_token, active_elems);
  HIP_CHECK(hipStreamSynchronize(stream));

  bool apw_enabled = false;
  size_t apw_window_bytes = 0;
  int apw_max_window_bytes = 0;
  int persisting_l2_cache_max_bytes = 0;
  int l2_cache_bytes = 0;
  std::string apw_error;
  if (args.mode == "arena_apw" || args.mode == "arena_apw_lds" ||
      args.mode == "arena_apw_tma") {
    configure_apw(stream, arena, active_elems * sizeof(float), args.apw_hit_ratio,
                  &apw_enabled, &apw_window_bytes, &apw_max_window_bytes,
                  &persisting_l2_cache_max_bytes, &l2_cache_bytes, &apw_error);
  }
  bool lds_tma_mode = args.mode == "arena_apw_lds" || args.mode == "arena_apw_tma";

  for (int i = 0; i < args.warmup; ++i) {
    launch_target(args, stream, full, indices, arena, out, sink, active_tokens,
                  elems_per_token, active_elems, grid_blocks);
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < args.timing_iters; ++i) {
    launch_target(args, stream, full, indices, arena, out, sink, active_tokens,
                  elems_per_token, active_elems, grid_blocks);
  }
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  elapsed_ms /= std::max(args.timing_iters, 1);

  std::cout << "{\n";
  std::cout << "  \"mode\": \"" << args.mode << "\",\n";
  std::cout << "  \"elapsed_ms\": " << elapsed_ms << ",\n";
  std::cout << "  \"active_tokens\": " << active_tokens << ",\n";
  std::cout << "  \"active_bytes\": " << active_elems * sizeof(float) << ",\n";
  std::cout << "  \"logical_bytes\": "
            << active_elems * sizeof(float) * size_t(args.passes) << ",\n";
  std::cout << "  \"apw_enabled\": " << (apw_enabled ? "true" : "false") << ",\n";
  std::cout << "  \"apw_window_bytes\": " << apw_window_bytes << ",\n";
  std::cout << "  \"apw_supported\": "
            << (apw_max_window_bytes > 0 ? "true" : "false") << ",\n";
  std::cout << "  \"apw_max_window_bytes\": " << apw_max_window_bytes << ",\n";
  std::cout << "  \"persisting_l2_cache_max_bytes\": "
            << persisting_l2_cache_max_bytes << ",\n";
  std::cout << "  \"l2_cache_bytes\": " << l2_cache_bytes << ",\n";
  std::cout << "  \"apw_error\": \"" << apw_error << "\",\n";
  std::cout << "  \"tma_analogue_backend\": \""
            << (lds_tma_mode ? "lds_tile_staging" : "none")
            << "\",\n";
  std::cout << "  \"lds_tile_bytes\": " << (lds_tma_mode ? args.lds_tile_bytes : 0)
            << ",\n";
  std::cout << "  \"lds_vector_width_bytes\": "
            << (lds_tma_mode ? args.lds_vector_width_bytes : 0) << ",\n";
  std::cout << "  \"amd_feature_path\": \""
            << (lds_tma_mode
                    ? (apw_enabled ? "hip_access_policy_window+lds_tile_staging_b128"
                                   : "apw_failed+lds_tile_staging_b128")
                    : (apw_enabled ? "hip_access_policy_window" : "none"))
            << "\",\n";
  std::cout << "  \"wavefront_size\": " << props.warpSize << ",\n";
  std::cout << "  \"compute_units\": " << props.multiProcessorCount << ",\n";
  std::cout << "  \"device\": \"" << props.name << "\"\n";
  std::cout << "}\n";

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(full));
  HIP_CHECK(hipFree(arena));
  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(sink));
  HIP_CHECK(hipFree(indices));
  return 0;
}
