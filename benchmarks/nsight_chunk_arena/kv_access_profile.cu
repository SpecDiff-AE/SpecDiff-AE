#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef DIFFSPEC_DISABLE_TMA_ASYNC
#include <cooperative_groups.h>
#include <cuda/barrier>
#endif

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "    \
                << cudaGetErrorString(_err) << std::endl;                     \
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
  int tma_tile_bytes = 16384;
  float apw_hit_ratio = 0.85f;
  unsigned int seed = 20260610u;
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
  read_arg(argc, argv, "--tma-tile-bytes", &args.tma_tile_bytes);
  read_arg(argc, argv, "--apw-hit-ratio", &args.apw_hit_ratio);
  read_arg(argc, argv, "--seed", &args.seed);
  return args;
}

__global__ void diffspec_init_linear_kernel(float* data, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x) {
    uint32_t x = static_cast<uint32_t>(i * 1664525ull + 1013904223ull);
    data[i] = static_cast<float>(x & 0xffffu) * 1.52587890625e-5f;
  }
}

__global__ void diffspec_stage_arena_kernel(const float* __restrict__ full,
                                            const int* __restrict__ indices,
                                            float* __restrict__ arena,
                                            int active_tokens,
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

__global__ void diffspec_baseline_sparse_kv_gather_kernel(
    const float* __restrict__ full,
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

__global__ void diffspec_chunk_arena_kernel(const float* __restrict__ arena,
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

#ifndef DIFFSPEC_DISABLE_TMA_ASYNC
__global__ void diffspec_chunk_arena_apw_tma_kernel(
    const float* __restrict__ arena,
    float* __restrict__ out,
    float* __restrict__ sink,
    size_t active_elems,
    int passes,
    int tile_elems) {
  namespace cg = cooperative_groups;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

  cg::thread_block block = cg::this_thread_block();
  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }
  __syncthreads();

  float* smem = reinterpret_cast<float*>(smem_raw);
  float local = 0.0f;
  size_t tiles = (active_elems + tile_elems - 1) / tile_elems;

  for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    size_t start = tile * size_t(tile_elems);
    int count = static_cast<int>(min(size_t(tile_elems), active_elems - start));
    int copy_bytes = count * static_cast<int>(sizeof(float));
    copy_bytes = (copy_bytes / 16) * 16;
    if (copy_bytes <= 0) {
      continue;
    }

    cuda::memcpy_async(
        block,
        smem,
        arena + start,
        cuda::aligned_size_t<16>(static_cast<size_t>(copy_bytes)),
        barrier);
    barrier.arrive_and_wait();

    int copy_elems = copy_bytes / static_cast<int>(sizeof(float));
    for (int local_idx = threadIdx.x; local_idx < copy_elems;
         local_idx += blockDim.x) {
      float acc = 0.0f;
      for (int pass = 0; pass < passes; ++pass) {
        int reuse_idx = (local_idx + pass * 257) % copy_elems;
        acc += smem[reuse_idx];
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
#endif

void launch_target(const Args& args,
                   cudaStream_t stream,
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
    diffspec_baseline_sparse_kv_gather_kernel<<<grid_blocks, args.block_threads, 0, stream>>>(
        full, indices, out, sink, active_tokens, elems_per_token, active_elems, args.passes);
  } else if (args.mode == "arena" || args.mode == "arena_apw") {
    diffspec_chunk_arena_kernel<<<grid_blocks, args.block_threads, 0, stream>>>(
        arena, out, sink, active_elems, args.passes);
  } else if (args.mode == "arena_apw_tma") {
#ifndef DIFFSPEC_DISABLE_TMA_ASYNC
    int tile_bytes = std::max(16, (args.tma_tile_bytes / 16) * 16);
    int tile_elems = tile_bytes / static_cast<int>(sizeof(float));
    diffspec_chunk_arena_apw_tma_kernel<<<grid_blocks, args.block_threads, tile_bytes, stream>>>(
        arena, out, sink, active_elems, args.passes, tile_elems);
#else
    diffspec_chunk_arena_kernel<<<grid_blocks, args.block_threads, 0, stream>>>(
        arena, out, sink, active_elems, args.passes);
#endif
  } else {
    std::cerr << "Unknown mode: " << args.mode << std::endl;
    std::exit(2);
  }
}

void configure_apw(cudaStream_t stream,
                   const void* ptr,
                   size_t bytes,
                   float hit_ratio,
                   bool* enabled,
                   size_t* window_bytes) {
  *enabled = false;
  *window_bytes = 0;
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int access_policy_max = 0;
  int persisting_max = 0;
  cudaError_t attr_err = cudaDeviceGetAttribute(
      &access_policy_max, cudaDevAttrMaxAccessPolicyWindowSize, device);
  cudaError_t l2_err = cudaDeviceGetAttribute(
      &persisting_max, cudaDevAttrMaxPersistingL2CacheSize, device);
  if (attr_err != cudaSuccess || l2_err != cudaSuccess ||
      access_policy_max <= 0 || persisting_max <= 0) {
    return;
  }

  size_t requested = std::min(bytes, static_cast<size_t>(access_policy_max));
  requested = std::max<size_t>(0, requested);
  size_t persisting = std::min(requested, static_cast<size_t>(persisting_max));
  if (persisting == 0) {
    return;
  }
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persisting));

  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
  attr.accessPolicyWindow.num_bytes = requested;
  attr.accessPolicyWindow.hitRatio = hit_ratio;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaError_t stream_err = cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &attr);
  if (stream_err == cudaSuccess) {
    *enabled = true;
    *window_bytes = requested;
  }
}

void clear_apw(cudaStream_t stream) {
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.num_bytes = 0;
  (void)cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
  (void)cudaCtxResetPersistingL2Cache();
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
  size_t full_bytes = full_elems * sizeof(float);
  size_t active_bytes = active_elems * sizeof(float);
  size_t logical_read_bytes = active_bytes * size_t(args.passes);

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  CUDA_CHECK(cudaSetDevice(0));

  float* d_full = nullptr;
  float* d_arena = nullptr;
  float* d_out = nullptr;
  float* d_sink = nullptr;
  int* d_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_full, full_bytes));
  CUDA_CHECK(cudaMalloc(&d_arena, active_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, active_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_indices, sizeof(int) * active_tokens));
  CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(float)));

  std::vector<int> indices = make_sparse_indices(args, active_tokens);
  CUDA_CHECK(cudaMemcpy(d_indices, indices.data(), sizeof(int) * active_tokens,
                        cudaMemcpyHostToDevice));

  int init_blocks = std::min<int>(65535, int((full_elems + 255) / 256));
  diffspec_init_linear_kernel<<<init_blocks, 256>>>(d_full, full_elems);
  CUDA_CHECK(cudaGetLastError());
  int stage_blocks = std::min<int>(65535, int((active_elems + 255) / 256));
  diffspec_stage_arena_kernel<<<stage_blocks, 256>>>(
      d_full, d_indices, d_arena, active_tokens, elems_per_token, active_elems);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  bool apw_enabled = false;
  size_t apw_window = 0;
  if (args.mode == "arena_apw" || args.mode == "arena_apw_tma") {
    configure_apw(stream, d_arena, active_bytes, args.apw_hit_ratio,
                  &apw_enabled, &apw_window);
  }

  int sm_count = props.multiProcessorCount;
  int grid_blocks = std::max(sm_count * 8, 1);

  for (int i = 0; i < args.warmup; ++i) {
    launch_target(args, stream, d_full, d_indices, d_arena, d_out, d_sink,
                  active_tokens, elems_per_token, active_elems, grid_blocks);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < args.timing_iters; ++i) {
    launch_target(args, stream, d_full, d_indices, d_arena, d_out, d_sink,
                  active_tokens, elems_per_token, active_elems, grid_blocks);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  elapsed_ms /= std::max(1, args.timing_iters);

  CUDA_CHECK(cudaProfilerStart());
  launch_target(args, stream, d_full, d_indices, d_arena, d_out, d_sink,
                active_tokens, elems_per_token, active_elems, grid_blocks);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaProfilerStop());

  float sink = 0.0f;
  CUDA_CHECK(cudaMemcpy(&sink, d_sink, sizeof(float), cudaMemcpyDeviceToHost));

#ifndef DIFFSPEC_DISABLE_TMA_ASYNC
  bool tma_compiled = true;
#else
  bool tma_compiled = false;
#endif

  double gb = double(logical_read_bytes) / 1.0e9;
  double gbps = gb / (double(elapsed_ms) / 1000.0);
  std::cout << "RESULT_JSON {"
            << "\"mode\":\"" << args.mode << "\","
            << "\"gpu\":\"" << props.name << "\","
            << "\"full_tokens\":" << args.full_tokens << ","
            << "\"active_tokens\":" << active_tokens << ","
            << "\"heads\":" << args.heads << ","
            << "\"head_dim\":" << args.head_dim << ","
            << "\"kv_parts\":" << args.kv_parts << ","
            << "\"passes\":" << args.passes << ","
            << "\"full_bytes\":" << full_bytes << ","
            << "\"active_bytes\":" << active_bytes << ","
            << "\"logical_read_bytes\":" << logical_read_bytes << ","
            << "\"elapsed_ms\":" << elapsed_ms << ","
            << "\"logical_read_gbps\":" << gbps << ","
            << "\"apw_enabled\":" << (apw_enabled ? "true" : "false") << ","
            << "\"apw_window_bytes\":" << apw_window << ","
            << "\"tma_async_compiled\":" << (tma_compiled ? "true" : "false") << ","
            << "\"sink\":" << sink
            << "}" << std::endl;

  if (apw_enabled) {
    clear_apw(stream);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_full));
  CUDA_CHECK(cudaFree(d_arena));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_indices));
  return 0;
}
