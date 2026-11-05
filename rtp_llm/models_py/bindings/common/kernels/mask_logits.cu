#include "rtp_llm/models_py/bindings/common/kernels/mask_logits.h"

#include <cfloat>

#if USING_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half
#include <cuda_bf16.h>  // For __nv_bfloat16
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

#if USING_CUDA
#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif
#ifndef CUDART_INF_BF16
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif
#endif

#if USING_ROCM
#ifndef HIP_INF_FP16
#define HIP_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif
#ifndef HIP_INF_BF16
#define HIP_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif
#define CUDART_INF_FP16 HIP_INF_FP16
#define CUDART_INF_BF16 HIP_INF_BF16
#endif

template<typename T>
__device__ T NegativeInfinity() {
    return -INFINITY;
}

template<>
__device__ __half NegativeInfinity<__half>() {
    return -CUDART_INF_FP16;
}

template<>
__device__ __nv_bfloat16 NegativeInfinity<__nv_bfloat16>() {
    return -CUDART_INF_BF16;
}

// Match BaseLogitsProcessor::neg_inf after conversion to the logits dtype.
template<typename T>
__device__ T MaskedLogitValue() {
    return -FLT_MAX;
}

template<>
__device__ __half MaskedLogitValue<__half>() {
    return NegativeInfinity<__half>();
}

template<>
__device__ __nv_bfloat16 MaskedLogitValue<__nv_bfloat16>() {
    return NegativeInfinity<__nv_bfloat16>();
}

// Batch version kernel for processing multiple beams
template<typename T>
__global__ void
mask_logits(const int batch_size, const int vocab_size, T* logits_batch, const uint8_t* __restrict__ mask_batch) {
    int batch_idx = blockIdx.y;
    int vocab_idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if (batch_idx < batch_size && vocab_idx < vocab_size) {
        int global_idx = batch_idx * vocab_size + vocab_idx;
        if (mask_batch[global_idx]) {
            logits_batch[global_idx] = NegativeInfinity<T>();
        }
    }
}

#if USING_CUDA
template<typename T>
__global__ void tree_sparse_mask(T* logits, const int32_t* controlled, int64_t controlled_count,
                                 const int32_t* candidates, const int64_t* offsets, const int32_t* rows,
                                 int64_t row_count, int64_t row_stride) {
    const int64_t total = row_count * controlled_count;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t row = index / controlled_count;
        if (rows[row] < 0) {
            continue;
        }
        const int32_t token = controlled[index % controlled_count];
        int64_t left = offsets[row];
        int64_t right = offsets[row + 1];
        const int64_t end = right;
        while (left < right) {
            const int64_t middle = left + (right - left) / 2;
            if (candidates[middle] < token) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        if (left == end || candidates[left] != token) {
            logits[static_cast<int64_t>(rows[row]) * row_stride + token] = NegativeInfinity<T>();
        }
    }
}

template<typename T>
void invokeTreeSparseMask(T* logits, const int32_t* controlled, int64_t controlled_count,
                          const int32_t* candidates, const int64_t* offsets, const int32_t* rows,
                          int64_t row_count, int64_t row_stride, cudaStream_t stream) {
    const int64_t blocks = (row_count * controlled_count + 255) / 256;
    if (blocks == 0) {
        return;
    }
    tree_sparse_mask<<<static_cast<unsigned>(blocks < 65535 ? blocks : 65535), 256, 0, stream>>>(
        logits, controlled, controlled_count, candidates, offsets, rows, row_count, row_stride);
    check_cuda_value(cudaPeekAtLastError());
}

template void invokeTreeSparseMask<float>(float*, const int32_t*, int64_t, const int32_t*, const int64_t*,
                                           const int32_t*, int64_t, int64_t, cudaStream_t);
template void invokeTreeSparseMask<half>(half*, const int32_t*, int64_t, const int32_t*, const int64_t*,
                                          const int32_t*, int64_t, int64_t, cudaStream_t);
template void invokeTreeSparseMask<__nv_bfloat16>(__nv_bfloat16*, const int32_t*, int64_t, const int32_t*,
                                                   const int64_t*, const int32_t*, int64_t, int64_t, cudaStream_t);
#endif

#if USING_CUDA
template<typename T>
__global__ void packed_mask_logits(T* __restrict__ logits_batch,
                                   const int32_t* __restrict__ packed_allow_mask,
                                   const int32_t* __restrict__ row_indices,
                                   const int mask_rows,
                                   const int logits_rows,
                                   const int logits_row_stride,
                                   const int vocab_size,
                                   const int bitmask_row_stride,
                                   const int bitmask_words) {
    const int compact_row = blockIdx.y;
    const int vocab_idx   = threadIdx.x + blockIdx.x * blockDim.x;
    if (compact_row >= mask_rows || vocab_idx >= vocab_size) {
        return;
    }

    const int logits_row = row_indices == nullptr ? compact_row : row_indices[compact_row];
    if (logits_row < 0 || logits_row >= logits_rows) {
        return;
    }

    const int word_idx = vocab_idx / 32;
    bool      allowed  = false;
    if (word_idx < bitmask_words) {
        const int64_t mask_offset =
            static_cast<int64_t>(compact_row) * static_cast<int64_t>(bitmask_row_stride) + word_idx;
        const uint32_t word = static_cast<uint32_t>(packed_allow_mask[mask_offset]);
        allowed             = (word & (1u << (vocab_idx % 32))) != 0u;
    }
    if (!allowed) {
        const int64_t logits_offset =
            static_cast<int64_t>(logits_row) * static_cast<int64_t>(logits_row_stride) + vocab_idx;
        logits_batch[logits_offset] = MaskedLogitValue<T>();
    }
}
#endif

template<typename T>
void invokeMaskLogits(T* logits_batch,
                      const uint8_t* __restrict__ mask_batch,
                      const int    batch_size,
                      const int    vocab_size,
                      cudaStream_t stream) {
    dim3 block, grid;

    block.x = 64;
    block.y = 1;
    block.z = 1;
    grid.x  = (vocab_size + block.x - 1) / block.x;
    grid.y  = batch_size;
    grid.z  = 1;

    mask_logits<<<grid, block, 0, stream>>>(batch_size, vocab_size, logits_batch, mask_batch);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

#if USING_CUDA
template<typename T>
void invokePackedMaskLogits(T* logits_batch,
                            const int32_t* __restrict__ packed_allow_mask,
                            const int32_t* __restrict__ row_indices,
                            const int    mask_rows,
                            const int    logits_rows,
                            const int    logits_row_stride,
                            const int    vocab_size,
                            const int    bitmask_row_stride,
                            const int    bitmask_words,
                            cudaStream_t stream) {
    const dim3 block(256);
    const dim3 grid((vocab_size + block.x - 1) / block.x, mask_rows);
    packed_mask_logits<<<grid, block, 0, stream>>>(logits_batch,
                                                   packed_allow_mask,
                                                   row_indices,
                                                   mask_rows,
                                                   logits_rows,
                                                   logits_row_stride,
                                                   vocab_size,
                                                   bitmask_row_stride,
                                                   bitmask_words);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}
#endif

template void invokeMaskLogits<float>(float* logits_batch,
                                      const uint8_t* __restrict__ mask_batch,
                                      const int    batch_size,
                                      const int    vocab_size,
                                      cudaStream_t stream);
template void invokeMaskLogits<half>(half* logits_batch,
                                     const uint8_t* __restrict__ mask_batch,
                                     const int    batch_size,
                                     const int    vocab_size,
                                     cudaStream_t stream);
template void invokeMaskLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                              const uint8_t* __restrict__ mask_batch,
                                              const int    batch_size,
                                              const int    vocab_size,
                                              cudaStream_t stream);

#if USING_CUDA
template void invokePackedMaskLogits<float>(float* logits_batch,
                                            const int32_t* __restrict__ packed_allow_mask,
                                            const int32_t* __restrict__ row_indices,
                                            const int    mask_rows,
                                            const int    logits_rows,
                                            const int    logits_row_stride,
                                            const int    vocab_size,
                                            const int    bitmask_row_stride,
                                            const int    bitmask_words,
                                            cudaStream_t stream);
template void invokePackedMaskLogits<half>(half* logits_batch,
                                           const int32_t* __restrict__ packed_allow_mask,
                                           const int32_t* __restrict__ row_indices,
                                           const int    mask_rows,
                                           const int    logits_rows,
                                           const int    logits_row_stride,
                                           const int    vocab_size,
                                           const int    bitmask_row_stride,
                                           const int    bitmask_words,
                                           cudaStream_t stream);
template void invokePackedMaskLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                                    const int32_t* __restrict__ packed_allow_mask,
                                                    const int32_t* __restrict__ row_indices,
                                                    const int    mask_rows,
                                                    const int    logits_rows,
                                                    const int    logits_row_stride,
                                                    const int    vocab_size,
                                                    const int    bitmask_row_stride,
                                                    const int    bitmask_words,
                                                    cudaStream_t stream);
#endif

}  // namespace rtp_llm
