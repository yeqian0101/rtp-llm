#include "gtest/gtest.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace rtp_llm {
namespace {
using Clock = std::chrono::steady_clock;
using Candidates = std::vector<std::vector<size_t>>;

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

std::string envValue(const char* name, const std::string& fallback) {
    const auto value = std::getenv(name);
    return value && *value ? value : fallback;
}

int positiveEnv(const char* name, int fallback) {
    const auto text = envValue(name, std::to_string(fallback));
    size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return value;
}

struct TempDictionary {
    std::string path;
    explicit TempDictionary(int controlled_count) {
        std::string pattern = envValue("TEST_TMPDIR", "/tmp") + "/tree-perf-XXXXXX";
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const int fd = mkstemp(buffer.data());
        if (fd < 0) {
            throw std::runtime_error("mkstemp failed");
        }
        close(fd);
        path = buffer.data();
        std::ofstream out(path);
        const auto sparse = envValue("TREE_PERF_SPARSE_MASK", "0");
        if (sparse != "0" && sparse != "1") {
            throw std::invalid_argument("TREE_PERF_SPARSE_MASK must be 0 or 1");
        }
        out << "{\"tree_decode_gpu_sparse_mask\":" << (sparse == "1" ? "true" : "false") << ",";
        out << R"("allow_unlisted_tokens":true,"start_token_id":1,"end_token_id":2,"prefix_dict":{"1":[)";
        for (int i = 0; i < controlled_count; ++i) {
            out << (i ? "," : "") << i + 10;
        }
        out << "]";
        for (int i = 0; i < controlled_count; ++i) {
            out << ",\"1_" << i + 10 << "\":[2]";
        }
        out << "}}";
        out.close();
        if (!out) {
            unlink(path.c_str());
            throw std::runtime_error("writing synthetic dictionary failed");
        }
    }
    ~TempDictionary() { unlink(path.c_str()); }
};

struct Events {
    cudaEvent_t begin{}, end{};
    Events() {
        checkCuda(cudaEventCreate(&begin));
        const auto result = cudaEventCreate(&end);
        if (result != cudaSuccess) {
            cudaEventDestroy(begin);
            checkCuda(result);
        }
    }
    ~Events() {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
};

double micros(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

void printStats(const std::string& phase, const char* metric, std::vector<double> values) {
    std::sort(values.begin(), values.end());
    auto percentile = [&](double p) { return values[static_cast<size_t>(std::ceil(p * values.size())) - 1]; };
    std::cout << std::fixed << std::setprecision(3) << "[tree-perf] phase=" << phase << " metric=" << metric
              << " mean_us=" << std::accumulate(values.begin(), values.end(), 0.0) / values.size()
              << " p50_us=" << percentile(0.50) << " p95_us=" << percentile(0.95)
              << " p99_us=" << percentile(0.99) << std::endl;
}

// Each sample starts with an idle stream. Wall includes completion, submit does
// not. Event intervals may include host enqueue gaps: they are NOT pure DMA time.
// Events are measured in a separate pass so event submission does not inflate wall.
void measure(const std::string& phase, const std::function<void()>& operation, bool gpu,
             cudaStream_t stream, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) {
        operation();
    }
    checkCuda(cudaStreamSynchronize(stream));
    std::vector<double> wall, submit, interval;
    for (int i = 0; i < iterations; ++i) {
        checkCuda(cudaStreamSynchronize(stream));
        const auto begin = Clock::now();
        operation();
        const auto enqueued = Clock::now();
        if (gpu) {
            checkCuda(cudaStreamSynchronize(stream));
        }
        wall.push_back(micros(begin, Clock::now()));
        submit.push_back(micros(begin, enqueued));
    }
    printStats(phase, "wall", wall);
    printStats(phase, "submit", submit);
    if (gpu) {
        Events events;
        for (int i = 0; i < iterations; ++i) {
            checkCuda(cudaEventRecord(events.begin, stream));
            operation();
            checkCuda(cudaEventRecord(events.end, stream));
            checkCuda(cudaEventSynchronize(events.end));
            float ms = 0;
            checkCuda(cudaEventElapsedTime(&ms, events.begin, events.end));
            interval.push_back(ms * 1000.0);
        }
        printStats(phase, "stream_interval", interval);
    }
}

Candidates collectCandidates(const std::vector<StreamTreeInfo>& infos) {
    Candidates result(infos.size());
    for (size_t i = 0; i < infos.size(); ++i) {
        // Match the production vector conversion and assignment.
        const auto& ids = infos[i].dfa_ptr->getCandidateTokenIds();
        result[i] = ids;
    }
    return result;
}

// Intentionally mirrors the current dense CPU path, not a proposed optimization.
torch::Tensor makeMask(const Candidates& candidates, size_t vocab, const PrefixToCandidateTokensPtr& prefix) {
    auto mask = torch::zeros({static_cast<int64_t>(candidates.size()), static_cast<int64_t>(vocab)}, torch::kUInt8);
    auto ptr = mask.data_ptr<uint8_t>();
    for (size_t row = 0; row < candidates.size(); ++row) {
        for (auto token : prefix->controlledTokens()) {
            ptr[row * vocab + token] = 1;
        }
        for (auto token : candidates[row]) {
            ptr[row * vocab + token] = 0;
        }
    }
    return mask;
}

TEST(TreeLogitsProcessorPerfTest, PhaseBreakdown) {
    torch::NoGradGuard no_grad;
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0) << "This benchmark requires CUDA; a skipped benchmark is not a measurement.";
    const int vocab = positiveEnv("TREE_PERF_VOCAB", 65536);
    const int warmup = positiveEnv("TREE_PERF_WARMUP", 20);
    const int iterations = positiveEnv("TREE_PERF_ITERS", 100);
    TempDictionary synthetic(positiveEnv("TREE_PERF_CONTROLLED", 2048));
    const auto external = envValue("TREE_PERF_DICT", "");
    const auto prefix = PrefixToCandidateTokens::instance();
    prefix->reloadPrefixDict(external.empty() ? synthetic.path : external);
    ASSERT_TRUE(prefix->initSuccess());
    ASSERT_TRUE(prefix->allowUnlistedTokens()) << "Use an allow_unlisted_tokens=true dictionary.";
    for (auto token : prefix->controlledTokens()) {
        ASSERT_GE(token, 0);
        ASSERT_LT(token, vocab);
    }
    const auto dtype_name = envValue("TREE_PERF_DTYPE", "fp16");
    ASSERT_TRUE(dtype_name == "fp16" || dtype_name == "fp32" || dtype_name == "bf16");
    const auto dtype = dtype_name == "fp32" ? torch::kFloat32 :
                       dtype_name == "bf16" ? torch::kBFloat16 : torch::kFloat16;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    int device = 0;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device));
    std::cout << "[tree-perf] gpu=" << properties.name << " vocab=" << vocab
              << " controlled=" << prefix->controlledTokens().size() << " dtype=" << dtype_name
              << " sparse=" << (prefix->gpuSparseMask() ? "true" : "false")
              << " warmup=" << warmup << " iterations=" << iterations
              << " dictionary=" << (external.empty() ? "synthetic" : external) << std::endl;

    std::vector<int> batches = {1, 4, 16};
    if (!envValue("TREE_PERF_BEAMS", "").empty()) {
        batches = {positiveEnv("TREE_PERF_BEAMS", 1)};
    }
    const std::string root = std::to_string(prefix->startTokenId());
    std::vector<std::string> states = external.empty() ? std::vector<std::string>{root, "1_10"} :
                                                      std::vector<std::string>{root};
    if (!envValue("TREE_PERF_STATES", "").empty()) {
        states.clear();
        std::istringstream input(envValue("TREE_PERF_STATES", ""));
        std::string state;
        while (std::getline(input, state, ',')) {
            ASSERT_TRUE(prefix->isValidStatus(state)) << state;
            states.push_back(state);
        }
        ASSERT_FALSE(states.empty());
    }

    for (int batch : batches) {
        // Separate scenarios for uniform root/deep state, then mixed beam states.
        for (size_t scenario = 0; scenario < states.size() + (states.size() > 1 ? 1 : 0); ++scenario) {
            std::vector<StreamTreeInfo> infos;
            for (int row = 0; row < batch; ++row) {
                auto dfa = std::make_shared<TreeDFA<std::string, int>>(prefix);
                dfa->forceSetStatus(states[scenario < states.size() ? scenario : row % states.size()]);
                infos.emplace_back(true, 0, 0, true, dfa);
            }
            TreeLogitsProcessor processor(infos);
            const auto candidates = collectCandidates(infos);
            size_t candidate_count = 0;
            for (const auto& row : candidates) {
                candidate_count += row.size();
                for (auto token : row) {
                    ASSERT_LT(token, static_cast<size_t>(vocab));
                }
            }
            const size_t bytes = static_cast<size_t>(batch) * vocab;
            std::cout << "[tree-perf] case beams=" << batch << " scenario=" << scenario
                      << " state=" << (scenario < states.size() ? states[scenario] : "mixed")
                      << " candidates_total=" << candidate_count << " mask_bytes=" << bytes
                      << " sparse_h2d_bytes=" << (candidate_count * sizeof(int32_t)
                          + (batch + 1) * sizeof(int64_t) + batch * sizeof(int32_t)) << std::endl;
            auto host_mask = makeMask(candidates, vocab, prefix);
            auto device_mask = host_mask.to(torch::Device(torch::kCUDA, device));
            auto copy_target = torch::empty_like(device_mask);
            auto pinned_mask =
                torch::empty(host_mask.sizes(), torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
            pinned_mask.copy_(host_mask);
            SamplerInputs inputs;
            inputs.logits =
                torch::ones({batch, vocab}, torch::TensorOptions().dtype(dtype).device(device_mask.device()));

            // Verify the fixed-state mask against the actual production processor
            // before timing. Repeated masking is idempotent; no reset/copy pollutes timing.
            ASSERT_FALSE(processor.process(inputs, 0, batch).has_value());
            auto reference = torch::ones_like(inputs.logits);
            processor.maskLogits(reference, device_mask);
            ASSERT_TRUE(torch::equal(inputs.logits, reference));
            auto cpu_logits = inputs.logits.to(torch::kCPU).to(torch::kFloat32);
            const auto values = cpu_logits.data_ptr<float>();
            const auto mask_values = host_mask.data_ptr<uint8_t>();
            for (size_t i = 0; i < bytes; ++i) {
                ASSERT_TRUE(mask_values[i] ? (std::isinf(values[i]) && values[i] < 0) : values[i] == 1.0f) << i;
            }

            auto bench = [&](const std::string& name, const std::function<void()>& op, bool gpu) {
                measure(name, op, gpu, stream, warmup, iterations);
            };
            Candidates collected;
            torch::Tensor generated, uploaded;
            volatile size_t validated = 0;
            bench("noop", [] {}, false);
            bench("candidates_cpu", [&] { collected = collectCandidates(infos); }, false);
            bench("validate_controlled_cpu", [&] {
                size_t count = 0;
                for (auto token : prefix->controlledTokens()) {
                    if (token < 0 || token >= vocab) {
                        throw std::runtime_error("controlled token out of range");
                    }
                    ++count;
                }
                validated = count;
            }, false);
            EXPECT_EQ(validated, prefix->controlledTokens().size());
            bench("mask_cpu", [&] { generated = makeMask(candidates, vocab, prefix); }, false);
            bench("upload_to_allocating", [&] { uploaded = host_mask.to(inputs.logits.device()); }, true);
            bench("h2d_pageable_preallocated", [&] {
                checkCuda(cudaMemcpyAsync(
                    copy_target.data_ptr(), host_mask.data_ptr(), bytes, cudaMemcpyHostToDevice, stream));
            }, true);
            ASSERT_TRUE(torch::equal(copy_target, device_mask));
            bench("h2d_pinned_preallocated", [&] {
                checkCuda(cudaMemcpyAsync(
                    copy_target.data_ptr(), pinned_mask.data_ptr(), bytes, cudaMemcpyHostToDevice, stream));
            }, true);
            ASSERT_TRUE(torch::equal(copy_target, device_mask));
            bench("resident_mask_kernel", [&] { processor.maskLogits(inputs.logits, device_mask); }, true);
            bench("prebuilt_cpu_upload_and_kernel", [&] {
                auto mask = host_mask.to(inputs.logits.device());
                processor.maskLogits(inputs.logits, mask);
            }, true);
            bench("full_process", [&] {
                if (processor.process(inputs, 0, batch).has_value()) {
                    throw std::runtime_error("TreeLogitsProcessor::process failed");
                }
            }, true);
            ASSERT_TRUE(torch::equal(inputs.logits, reference));
        }
    }
}
}  // namespace
}  // namespace rtp_llm
