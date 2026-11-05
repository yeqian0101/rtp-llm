#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <cstring>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/common/kernels/activation_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/mask_logits.h"

namespace rtp_llm {

void cudaSoftmaxInplace(torch::Tensor& input, cudaStream_t stream) {
    RTP_LLM_CHECK(input.dim() == 2);
    RTP_LLM_CHECK(input.is_contiguous());
    int  m     = input.size(0);
    int  n     = input.size(1);
    auto dtype = input.scalar_type();
    if (dtype == torch::kFloat32) {
        invokeAddBiasSoftMax<float>(input.data_ptr<float>(), nullptr, nullptr, nullptr, m, n, n, stream);
    } else if (dtype == torch::kFloat16) {
        invokeAddBiasSoftMax<half>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()), nullptr, nullptr, nullptr, m, n, n, stream);
    } else if (dtype == torch::kBFloat16) {
        invokeAddBiasSoftMax<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            m,
                                            n,
                                            n,
                                            stream);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "cudaSoftmaxInplace: unsupported dtype");
    }
}

void cudaMaskLogits(torch::Tensor& logits, const torch::Tensor& mask, cudaStream_t stream) {
    RTP_LLM_CHECK(logits.dim() == 2);
    RTP_LLM_CHECK(mask.dim() == 2);
    RTP_LLM_CHECK(logits.size(0) == mask.size(0));
    RTP_LLM_CHECK(logits.size(1) == mask.size(1));
    size_t batch_size = logits.size(0);
    size_t vocab_size = logits.size(1);
    auto   dtype      = logits.scalar_type();
    if (dtype == torch::kFloat32) {
        invokeMaskLogits<float>(logits.data_ptr<float>(), mask.data_ptr<uint8_t>(), batch_size, vocab_size, stream);
    } else if (dtype == torch::kFloat16) {
        invokeMaskLogits<half>(reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
                               mask.data_ptr<uint8_t>(),
                               batch_size,
                               vocab_size,
                               stream);
    } else if (dtype == torch::kBFloat16) {
        invokeMaskLogits<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
                                        mask.data_ptr<uint8_t>(),
                                        batch_size,
                                        vocab_size,
                                        stream);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "cudaMaskLogits: unsupported dtype");
    }
}

namespace {
struct TreeControlledCache {
    torch::Tensor ids;
};

// Retain only the latest dictionary generation. Live processors keep their old
// cache alive until all their kernels complete. Reload is a quiescent operation.
std::shared_ptr<TreeControlledCache> treeControlledCache(
    uint64_t generation, const std::vector<int32_t>& controlled, const torch::Tensor& logits) {
    static std::mutex mutex;
    static uint64_t cached_generation = 0;
    static std::map<std::pair<int, int64_t>, std::shared_ptr<TreeControlledCache>> caches;
    std::lock_guard<std::mutex> lock(mutex);
    if (cached_generation != generation) {
        caches.clear();
        cached_generation = generation;
    }
    const auto key = std::make_pair(static_cast<int>(logits.get_device()), logits.size(1));
    auto& cache = caches[key];
    if (!cache) {
        auto created = std::make_shared<TreeControlledCache>();
        auto host = torch::empty({static_cast<int64_t>(controlled.size())}, torch::kInt32);
        std::memcpy(host.data_ptr<int32_t>(), controlled.data(), controlled.size() * sizeof(int32_t));
        created->ids = host.to(logits.device());
        // One-time initialization barrier permits sharing across CUDA streams.
        C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));
        cache = std::move(created);
    }
    return cache;
}

struct TreeSparseSlot {
    torch::Tensor host_ids, device_ids, host_offsets, device_offsets, host_rows, device_rows;
    at::cuda::CUDAEvent done;
    bool pending = false;

    void wait() {
        if (pending) {
            done.synchronize();
            pending = false;
        }
    }
    ~TreeSparseSlot() {
        // Includes the H2D source and the kernel inputs, even if the processor is
        // destroyed immediately after enqueue. Destructors must not throw.
        if (pending) {
            try {
                done.synchronize();
            } catch (const std::exception& error) {
                RTP_LLM_LOG_WARNING("tree sparse buffer completion failed: %s", error.what());
            }
        }
    }
    void reserve(torch::Tensor& host, torch::Tensor& device, int64_t count,
                 torch::ScalarType dtype, torch::Device gpu) {
        if (!host.defined() || host.numel() < count) {
            const int64_t capacity = std::max<int64_t>(count, host.defined() ? host.numel() * 2 : 64);
            host = torch::empty({capacity}, torch::TensorOptions().dtype(dtype).pinned_memory(true));
            device = torch::empty({capacity}, torch::TensorOptions().dtype(dtype).device(gpu));
        }
    }
};
}  // namespace

struct TreeSparseMaskState {
    uint64_t generation = 0;
    int device = -1;
    int64_t vocab = 0;
    std::shared_ptr<TreeControlledCache> controlled;
    // Slots are destroyed before the shared controlled tensor is released.
    std::array<TreeSparseSlot, 2> slots;
    size_t next_slot = 0;
};

void cudaTreeSparseMask(torch::Tensor& logits, uint64_t generation,
                        const std::vector<int32_t>& controlled,
                        const std::vector<const std::vector<int32_t>*>& candidates,
                        std::shared_ptr<TreeSparseMaskState>& state) {
    const c10::cuda::CUDAGuard guard(logits.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (!state || state->generation != generation || state->device != logits.get_device()
        || state->vocab != logits.size(1)) {
        auto created = std::make_shared<TreeSparseMaskState>();
        created->generation = generation;
        created->device = logits.get_device();
        created->vocab = logits.size(1);
        created->controlled = treeControlledCache(generation, controlled, logits);
        state = std::move(created);
        RTP_LLM_LOG_INFO("tree decode backend=gpu_sparse, generation=%llu, device=%d, vocab=%lld, controlled=%zu",
                         static_cast<unsigned long long>(generation), logits.get_device(),
                         static_cast<long long>(logits.size(1)), controlled.size());
    }
    auto& slot = state->slots[state->next_slot++ % state->slots.size()];
    slot.wait();
    int64_t count = 0;
    for (const auto* ids : candidates) {
        if (ids) {
            count += ids->size();
        }
    }
    const int64_t rows = candidates.size();
    slot.reserve(slot.host_ids, slot.device_ids, count, torch::kInt32, logits.device());
    slot.reserve(slot.host_offsets, slot.device_offsets, rows + 1, torch::kInt64, logits.device());
    slot.reserve(slot.host_rows, slot.device_rows, rows, torch::kInt32, logits.device());
    auto* ids = slot.host_ids.data_ptr<int32_t>();
    auto* offsets = slot.host_offsets.data_ptr<int64_t>();
    auto* row_ids = slot.host_rows.data_ptr<int32_t>();
    offsets[0] = 0;
    for (int64_t row = 0; row < rows; ++row) {
        row_ids[row] = candidates[row] ? static_cast<int32_t>(row) : -1;
        offsets[row + 1] = offsets[row];
        if (candidates[row]) {
            const auto& source = *candidates[row];
            std::memcpy(ids + offsets[row], source.data(), source.size() * sizeof(int32_t));
            offsets[row + 1] += source.size();
        }
    }
    // Record completion even if a later enqueue fails, so destruction/reuse
    // cannot release pinned sources while a successful earlier copy is pending.
    auto record = [&] {
        slot.done.record(stream);
        slot.pending = true;
    };
    try {
        C10_CUDA_CHECK(cudaMemcpyAsync(slot.device_ids.data_ptr(), ids, count * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream.stream()));
        C10_CUDA_CHECK(cudaMemcpyAsync(slot.device_offsets.data_ptr(), offsets, (rows + 1) * sizeof(int64_t),
                                      cudaMemcpyHostToDevice, stream.stream()));
        C10_CUDA_CHECK(cudaMemcpyAsync(slot.device_rows.data_ptr(), row_ids, rows * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream.stream()));
        const auto* c = state->controlled->ids.data_ptr<int32_t>();
        const auto* a = slot.device_ids.data_ptr<int32_t>();
        const auto* o = slot.device_offsets.data_ptr<int64_t>();
        const auto* r = slot.device_rows.data_ptr<int32_t>();
        const auto n = static_cast<int64_t>(controlled.size());
        if (logits.scalar_type() == torch::kFloat32) {
            invokeTreeSparseMask(logits.data_ptr<float>(), c, n, a, o, r, rows, logits.stride(0), stream.stream());
        } else if (logits.scalar_type() == torch::kFloat16) {
            invokeTreeSparseMask(reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
                                 c, n, a, o, r, rows, logits.stride(0), stream.stream());
        } else {
            invokeTreeSparseMask(reinterpret_cast<__nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
                                 c, n, a, o, r, rows, logits.stride(0), stream.stream());
        }
        record();
    } catch (...) {
        // Synchronize on an exceptional path, including event-record failure.
        cudaStreamSynchronize(stream.stream());
        slot.pending = false;
        throw;
    }
}

}  // namespace rtp_llm
