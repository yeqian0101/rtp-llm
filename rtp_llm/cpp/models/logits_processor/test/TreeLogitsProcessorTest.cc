
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include <unistd.h>
#include <fstream>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

using namespace std;

namespace rtp_llm {

class SamplerDataBuilder {
public:
    SamplerDataBuilder() = default;

    struct Config {
        size_t            batch_size;
        size_t            vocab_size;
        size_t            max_length;
        rtp_llm::DataType logits_type = rtp_llm::DataType::TYPE_FP32;
    };

    BaseLogitsProcessorPtr generateLogitsProcessor(bool in_tree_mode, size_t batch_size, std::string file_path) {
        std::vector<StreamTreeInfo> tree_infos;

        PrefixToCandidateTokens::instance()->reloadPrefixDict(file_path);

        for (size_t i = 0; i < batch_size; i++) {
            auto tree_info =
                StreamTreeInfo(in_tree_mode,
                               0,
                               0,
                               0,
                               std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()));
            tree_infos.push_back(tree_info);
        }

        BaseLogitsProcessorPtr processor_ptr = std::make_shared<TreeLogitsProcessor>(tree_infos);
        return processor_ptr;
    }

    SamplerInputs allocate(Config config, std::vector<BaseLogitsProcessorPtr> processors, std::vector<size_t> nums) {
        SamplerInputs sampler_inputs;

        sampler_inputs.step                = config.max_length;
        sampler_inputs.batch_size          = config.batch_size;
        sampler_inputs.batch_size_out      = config.batch_size;
        sampler_inputs.vocab_size          = config.vocab_size;
        LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
        for (size_t i = 0, idx = 0; i < processors.size(); i++) {
            state_ptr->insert(processors[i], idx, idx + nums[i]);
            idx += nums[i];
        }
        sampler_inputs.logits_processor_states_ptr = state_ptr;
        sampler_inputs.logits =
            torch::empty({(int64_t)config.batch_size, (int64_t)config.vocab_size},
                         torch::TensorOptions().dtype(dataTypeToTorchType(config.logits_type)).device(torch::kCUDA));
        sampler_inputs.sequence_lengths   = torch::empty({(int64_t)config.batch_size}, torch::kInt32);
        sampler_inputs.input_lengths      = torch::empty({(int64_t)config.batch_size}, torch::kInt32);
        sampler_inputs.num_beams_in       = torch::empty({(int64_t)config.batch_size}, torch::kLong);
        sampler_inputs.num_beams_out      = torch::empty({(int64_t)config.batch_size}, torch::kLong);
        sampler_inputs.top_k              = torch::empty({(int64_t)config.batch_size}, torch::kInt32);
        sampler_inputs.top_p              = torch::empty({(int64_t)config.batch_size}, torch::kFloat32);
        sampler_inputs.temperature        = torch::empty({(int64_t)config.batch_size}, torch::kFloat32);
        sampler_inputs.repetition_penalty = torch::empty({(int64_t)config.batch_size}, torch::kFloat32);
        sampler_inputs.cum_log_probs      = torch::empty({(int64_t)config.batch_size}, torch::kFloat32);
        sampler_inputs.token_ids =
            torch::empty({(int64_t)config.batch_size, (int64_t)(sampler_inputs.step + 1)}, torch::kInt32);
        sampler_inputs.logits.zero_();
        sampler_inputs.token_ids.zero_();
        return sampler_inputs;
    };

    void setTokenIds(SamplerInputs& sampler_inputs, std::vector<std::vector<int>>& token_ids) {
        RTP_LLM_CHECK(token_ids.size() == sampler_inputs.batch_size);
        RTP_LLM_CHECK(token_ids[0].size() == sampler_inputs.step + 1);
        for (auto i = 0; i < sampler_inputs.batch_size; i++) {
            auto tensor = sampler_inputs.token_ids[i];
            for (auto j = 0; j < sampler_inputs.step + 1; j++) {
                tensor[j] = token_ids[i][j];
            }
        }
    }
};

class TreeLogitsProcessorTest: public DeviceTestBase {};

#define EXPECT_SIMILAR(vec1, vec2, eps)                                                                                \
    do {                                                                                                               \
        bool similar = true;                                                                                           \
        if (vec1.size() != vec2.size()) {                                                                              \
            similar = false;                                                                                           \
        } else {                                                                                                       \
            for (size_t i = 0; i < vec1.size(); ++i) {                                                                 \
                if (std::fabs(vec1[i] - vec2[i]) >= eps) {                                                             \
                    similar = false;                                                                                   \
                    break;                                                                                             \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        EXPECT_TRUE(similar) << "Vectors are not similar";                                                             \
    } while (0)

TEST_F(TreeLogitsProcessorTest, testGenerateVocabMask) {
    SamplerDataBuilder     builder;
    std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_dict.json";
    size_t                 batch_size = 4;
    size_t                 vocab_size = 1024;
    size_t                 max_length = 1024;
    BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
    SamplerInputs sampler_inputs = builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});
    std::vector<std::vector<size_t>> batch_candidate_token_ids = {{}, {1}, {2, 3, 4}, {1, 3, 5}};
    auto vocab_mask     = processor->generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
    auto vocab_mask_cpu = vocab_mask.cpu().contiguous();

    std::vector<std::vector<int32_t>> expect_vocab_mask(batch_size, std::vector<int32_t>(vocab_size, 1));
    expect_vocab_mask[1][1] = 0;
    expect_vocab_mask[2][2] = 0;
    expect_vocab_mask[2][3] = 0;
    expect_vocab_mask[2][4] = 0;
    expect_vocab_mask[3][1] = 0;
    expect_vocab_mask[3][3] = 0;
    expect_vocab_mask[3][5] = 0;

    auto vocab_mask_hosts = vocab_mask_cpu.data_ptr<uint8_t>();
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            ASSERT_TRUE(vocab_mask_hosts[i * vocab_size + j] == expect_vocab_mask[i][j]);
        }
    }
}

template<typename Dtype>
void setTensor(torch::Tensor& tensor, std::vector<std::vector<Dtype>> content) {
    RTP_LLM_CHECK(tensor.dim() == 2);
    RTP_LLM_CHECK((size_t)tensor.size(0) == content.size());
    RTP_LLM_CHECK((size_t)tensor.size(1) == content[0].size());
    for (int64_t i = 0; i < tensor.size(0); i++) {
        for (int64_t j = 0; j < tensor.size(1); j++) {
            tensor[i][j] = content[i][j];
        }
    }
}

TEST_F(TreeLogitsProcessorTest, testUpdateStatus) {
    {
        SamplerDataBuilder     builder;
        std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_dict.json";
        size_t                 batch_size = 4;
        size_t                 vocab_size = 1024;
        size_t                 max_length = 10;
        BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
        SamplerInputs          sampler_inputs =
            builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});

        auto new_token = torch::tensor({{64000}, {64003}, {64006}, {64008}}, torch::kInt32);

        processor->updateStatus(new_token, 1);

        auto                     proc        = std::dynamic_pointer_cast<TreeLogitsProcessor>(processor);
        std::vector<std::string> status_list = proc->getStatus();
        EXPECT_EQ("225_64000", status_list[0]);
        EXPECT_EQ("225_64003", status_list[1]);
        EXPECT_EQ("225_64006", status_list[2]);
        EXPECT_EQ("225_64008", status_list[3]);

        std::vector<std::vector<int>> token_ids_2 = {{64001}, {64001}, {64004}, {64001}};
        setTensor(new_token, token_ids_2);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001", status_list[0]);
        EXPECT_EQ("225_64003_64001", status_list[1]);
        EXPECT_EQ("225_64006_64004", status_list[2]);
        EXPECT_EQ("225_64008_64001", status_list[3]);

        std::vector<std::vector<int>> token_ids_3 = {{2}, {2}, {2}, {2}};
        setTensor(new_token, token_ids_3);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001_2", status_list[0]);
        EXPECT_EQ("225_64003_64001_2", status_list[1]);
        EXPECT_EQ("225_64006_64004_2", status_list[2]);
        EXPECT_EQ("225_64008_64001_2", status_list[3]);

        std::vector<std::vector<int>> token_ids_4 = {{1}, {1}, {1}, {1}};
        setTensor(new_token, token_ids_4);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001_2", status_list[0]);
        EXPECT_EQ("225_64003_64001_2", status_list[1]);
        EXPECT_EQ("225_64006_64004_2", status_list[2]);
        EXPECT_EQ("225_64008_64001_2", status_list[3]);
    }
}

TEST_F(TreeLogitsProcessorTest, testProcess) {
    {
        SamplerDataBuilder     builder;
        std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_dict.json";
        size_t                 batch_size = 4;
        size_t                 vocab_size = 100000;
        size_t                 max_length = 10;
        BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
        SamplerInputs          sampler_inputs =
            builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});

        std::vector<std::vector<int>> token_ids = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9}};
        builder.setTokenIds(sampler_inputs, token_ids);

        std::vector<std::vector<float>> logits_list;
        std::vector<std::vector<float>> logits_index_list = {{64000}, {64003, 64006, 64008}, {64011}, {64001}};
        for (size_t i = 0; i < batch_size; i++) {
            auto tensor = sampler_inputs.logits[i];
            tensor.fill_(0);
            for (auto index : logits_index_list[i]) {
                tensor[index] = 1;
            }
        }
        processor->process(sampler_inputs, 0, batch_size);

        auto logits_tensor = sampler_inputs.logits[0].cpu();
        auto logits_hosts  = std::vector<float>(logits_tensor.data_ptr<float>(),
                                               logits_tensor.data_ptr<float>() + logits_tensor.numel());
        ASSERT_EQ(logits_hosts[64000], 1);
        ASSERT_EQ(logits_hosts[64003], 0);
        ASSERT_EQ(logits_hosts[64011], 0);
        ASSERT_TRUE(logits_hosts[64001] == -INFINITY);
    }
}


class UnlistedTreeLogitsProcessorTest: public DeviceTestBase, public ::testing::WithParamInterface<bool> {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        path_ = "/tmp/unlisted_tree_logits_" + std::to_string(getpid()) + ".json";
        load("true");
    }
    void TearDown() override {
        std::remove(path_.c_str());
        DeviceTestBase::TearDown();
    }
    void load(const std::string& mode) { load(mode, GetParam()); }
    void load(const std::string& mode, bool sparse) {
        std::ofstream file(path_);
        file << "{\"tree_decode_gpu_sparse_mask\":" << (sparse ? "true" : "false") << ",";
        if (!mode.empty()) {
            file << "\"allow_unlisted_tokens\":" << mode << ",";
        }
        file << R"("start_token_id":1,"end_token_id":2,"prefix_dict":{
            "1":[10,20],"1_10":[30],"1_10_30":[],"1_20":[40],"1_20_40":[2]}})";
        file.close();
        PrefixToCandidateTokens::instance()->reloadPrefixDict(path_);
        ASSERT_TRUE(PrefixToCandidateTokens::instance()->initSuccess());
    }
    StreamTreeInfo info(const std::string& state = "1", bool active = true, bool beam = false, int input_length = 0) {
        auto dfa = std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance());
        dfa->forceSetStatus(state);
        return StreamTreeInfo(active, input_length, 0, beam, dfa);
    }
    torch::Tensor tokens(const std::vector<int>& values, int rows) {
        return torch::tensor(values, torch::kInt32).reshape({rows, static_cast<int64_t>(values.size() / rows)});
    }
    // Check every vocabulary entry, including all ordinary tokens and unchanged rows.
    void expectRow(const torch::Tensor& actual,
                   const torch::Tensor& before,
                   int row,
                   const std::unordered_set<int>& blocked) {
        const auto a = actual.cpu().to(torch::kFloat32).contiguous();
        const auto b = before.cpu().to(torch::kFloat32).contiguous();
        for (int token = 0; token < actual.size(1); ++token) {
            SCOPED_TRACE("row=" + std::to_string(row) + " token=" + std::to_string(token));
            const auto index = row * actual.size(1) + token;
            if (blocked.count(token)) {
                EXPECT_EQ(-INFINITY, a.data_ptr<float>()[index]);
            } else {
                EXPECT_EQ(b.data_ptr<float>()[index], a.data_ptr<float>()[index]);
            }
        }
    }
    std::string path_;
};

TEST_P(UnlistedTreeLogitsProcessorTest, MasksOnlyControlledTokensAcrossDtypesAndSubranges) {
    for (auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        TreeLogitsProcessor processor({info("1"), info("1_10"), info("1", false)});
        SamplerInputs inputs;
        inputs.logits = torch::arange(5 * 64, torch::kFloat32).reshape({5, 64}).to(dtype).to(torch::kCUDA);
        const auto before = inputs.logits.clone();
        ASSERT_FALSE(processor.process(inputs, 1, 4).has_value());
        expectRow(inputs.logits, before, 0, {});
        expectRow(inputs.logits, before, 1, {2, 30, 40});
        expectRow(inputs.logits, before, 2, {2, 10, 20, 40});
        expectRow(inputs.logits, before, 3, {}); // Inactive row in an active batch.
        expectRow(inputs.logits, before, 4, {});
    }
}

TEST_P(UnlistedTreeLogitsProcessorTest, ResetChangesNextStepMask) {
    TreeLogitsProcessor processor({info()});
    ASSERT_FALSE(processor.updateStatus(tokens({10}, 1), 1).has_value());
    SamplerInputs inputs;
    inputs.logits = torch::ones({1, 64}, torch::TensorOptions().device(torch::kCUDA));
    auto before = inputs.logits.clone();
    ASSERT_FALSE(processor.process(inputs, 0, 1).has_value());
    expectRow(inputs.logits, before, 0, {2, 10, 20, 40});
    ASSERT_FALSE(processor.updateStatus(tokens({9}, 1), 1).has_value());
    EXPECT_EQ((std::vector<std::string>{"1"}), processor.getStatus());
    inputs.logits = before.clone();
    ASSERT_FALSE(processor.process(inputs, 0, 1).has_value());
    expectRow(inputs.logits, before, 0, {2, 30, 40});
}

TEST_P(UnlistedTreeLogitsProcessorTest, BeamCopiesResetIndependentlyAndKeepOutputOffsets) {
    TreeLogitsProcessor processor({info("1", true, true, 2)});
    ASSERT_FALSE(processor.updateStatus(tokens({7, 8, 10}, 1), 1).has_value());
    processor.updateMultiSeqStatus({0, 0});
    ASSERT_FALSE(processor.updateStatus(tokens({7, 8, 10, 9, 7, 8, 10, 30}, 2), 1).has_value());
    EXPECT_EQ((std::vector<std::string>{"1", "1_10_30"}), processor.getStatus());
    // Reorder parents before the next step. Ordinary tokens still count in the
    // full output offset even though they clear the controlled-token path.
    processor.updateMultiSeqStatus({1, 0});
    ASSERT_FALSE(processor.updateStatus(tokens({7, 8, 10, 30, 2, 7, 8, 10, 9, 20}, 2), 1).has_value());
    EXPECT_EQ((std::vector<std::string>{"1_10_30_2", "1_20"}), processor.getStatus());
    SamplerInputs inputs;
    inputs.logits = torch::ones({2, 64}, torch::TensorOptions().device(torch::kCUDA));
    const auto before = inputs.logits.clone();
    ASSERT_FALSE(processor.process(inputs, 0, 2).has_value());
    expectRow(inputs.logits, before, 0, {10, 20, 30, 40});
    expectRow(inputs.logits, before, 1, {2, 10, 20, 30});
}

TEST_P(UnlistedTreeLogitsProcessorTest, MultiTokenUpdateRestartsWithinOneBatch) {
    TreeLogitsProcessor processor({info()});
    ASSERT_FALSE(processor.updateStatus(tokens({10, 9, 20, 40}, 1), 4).has_value());
    EXPECT_EQ((std::vector<std::string>{"1_20_40"}), processor.getStatus());
}

TEST_P(UnlistedTreeLogitsProcessorTest, EmptyMissingAndFinishedStatesKeepOrdinaryTokens) {
    auto finished = info("1_20_40");
    finished.dfa_ptr->next(2);
    TreeLogitsProcessor processor({info("1_10_30"), info("missing"), finished});
    SamplerInputs inputs;
    inputs.logits = torch::ones({3, 64}, torch::TensorOptions().device(torch::kCUDA));
    const auto before = inputs.logits.clone();
    ASSERT_FALSE(processor.process(inputs, 0, 3).has_value());
    for (int row = 0; row < 3; ++row) {
        expectRow(inputs.logits, before, row, {10, 20, 30, 40});
    }
    ASSERT_FALSE(processor.updateStatus(tokens({9, 9, 9}, 3), 1).has_value());
    EXPECT_EQ((std::vector<std::string>{"1", "1", "1"}), processor.getStatus());
}

TEST_P(UnlistedTreeLogitsProcessorTest, DisabledAndMissingSwitchKeepStrictWhitelist) {
    for (const std::string mode : {"false", ""}) {
        load(mode);
        TreeLogitsProcessor processor({info()});
        SamplerInputs inputs;
        inputs.logits = torch::ones({1, 64}, torch::TensorOptions().device(torch::kCUDA));
        const auto before = inputs.logits.clone();
        ASSERT_FALSE(processor.process(inputs, 0, 1).has_value());
        std::unordered_set<int> blocked;
        for (int token = 0; token < 64; ++token) {
            if (token != 10 && token != 20) {
                blocked.insert(token);
            }
        }
        expectRow(inputs.logits, before, 0, blocked);
    }
}

TEST_P(UnlistedTreeLogitsProcessorTest, OutOfVocabularyConfigurationReturnsErrorWithoutMasking) {
    TreeLogitsProcessor processor({info()});
    SamplerInputs inputs;
    inputs.logits = torch::ones({1, 32}, torch::TensorOptions().device(torch::kCUDA));
    const auto before = inputs.logits.clone();
    // 40 is controlled but out of range, even though it is not a root candidate.
    EXPECT_TRUE(processor.process(inputs, 0, 1).has_value());
    EXPECT_TRUE(torch::equal(before, inputs.logits));
}

TEST_P(UnlistedTreeLogitsProcessorTest, InactiveProcessorLeavesAllLogitsUnchanged) {
    TreeLogitsProcessor processor({info("1", false)});
    SamplerInputs inputs;
    inputs.logits = torch::ones({1, 64}, torch::TensorOptions().device(torch::kCUDA));
    const auto before = inputs.logits.clone();
    ASSERT_FALSE(processor.process(inputs, 0, 1).has_value());
    EXPECT_TRUE(torch::equal(before, inputs.logits));
}


TEST_P(UnlistedTreeLogitsProcessorTest, OrdinaryTokenCanWinGreedySelection) {
    TreeLogitsProcessor processor({info()});
    SamplerInputs inputs;
    inputs.logits = torch::zeros({1, 64}, torch::TensorOptions().device(torch::kCUDA));
    inputs.logits[0][30] = 100; // Controlled, but not a root candidate.
    inputs.logits[0][9] = 90;   // Ordinary token must remain selectable.
    inputs.logits[0][10] = 80;
    ASSERT_FALSE(processor.process(inputs, 0, 1).has_value());
    EXPECT_EQ(9, inputs.logits.argmax(1).item<int64_t>());
    const auto probs = torch::softmax(inputs.logits, 1).cpu();
    EXPECT_EQ(0.0f, probs[0][30].item<float>());
    EXPECT_GT(probs[0][9].item<float>(), 0.0f);
}

TEST_P(UnlistedTreeLogitsProcessorTest, BufferReuseAndReloadPreserveUpstreamMasks) {
    TreeLogitsProcessor processor({info(), info("1_10")});
    for (int step = 0; step < 12; ++step) {
        if (step == 6) {
            load("true"); // Existing processor must invalidate its GPU cache.
        }
        const int rows = step % 2 ? 3 : 2;
        processor.updateMultiSeqStatus(rows == 3 ? std::vector<int>{0, 1, 0} : std::vector<int>{0, 1});
        SamplerInputs inputs;
        inputs.logits = torch::randn({rows, 64}, torch::TensorOptions().device(torch::kCUDA));
        inputs.logits[0][10] = -INFINITY; // Legal here, but forbidden by an upstream processor.
        inputs.logits[0][9] = -INFINITY;  // Ordinary token must also remain forbidden.
        auto before = inputs.logits.clone();
        ASSERT_FALSE(processor.process(inputs, 0, rows).has_value());
        expectRow(inputs.logits, before, 0, {2, 30, 40});
        expectRow(inputs.logits, before, 1, {2, 10, 20, 40});
        if (rows == 3) {
            expectRow(inputs.logits, before, 2, {2, 30, 40});
        }
    }
}

TEST_P(UnlistedTreeLogitsProcessorTest, SparseCrossStreamReuseAndStridedRows) {
    if (!GetParam()) {
        GTEST_SKIP() << "Noncontiguous rows are an additional sparse-path capability";
    }
    auto moving = info();
    TreeLogitsProcessor processor({moving, info("1_10"), info("1", false)});
    auto other = info();
    TreeLogitsProcessor second({other, info("1_10"), info("1", false)});
    const auto first_stream = at::cuda::getStreamFromPool();
    const auto second_stream = at::cuda::getStreamFromPool();
    std::vector<torch::Tensor> actual, expected;
    // Do not synchronize between calls: both slots must protect in-flight data,
    // and two processors share the controlled cache across streams.
    for (int step = 0; step < 10; ++step) {
        // Vary data each time a ring slot is reused, so premature overwrites
        // cannot pass unnoticed simply because all uploads contain equal IDs.
        moving.dfa_ptr->forceSetStatus(step % 3 == 0 ? "1" : "1_20");
        other.dfa_ptr->forceSetStatus(step % 3 == 0 ? "1" : "1_20");
        const c10::cuda::CUDAStreamGuard guard(step % 2 ? first_stream : second_stream);
        SamplerInputs inputs;
        auto backing = torch::randn({5, 80}, torch::TensorOptions().device(torch::kCUDA));
        inputs.logits = backing.narrow(1, 0, 64);
        auto before = inputs.logits.clone();
        ASSERT_FALSE(processor.process(inputs, 1, 4).has_value());
        ASSERT_FALSE(second.process(inputs, 1, 4).has_value());
        actual.push_back(inputs.logits);
        expected.push_back(before);
    }
    first_stream.synchronize();
    second_stream.synchronize();
    for (size_t step = 0; step < actual.size(); ++step) {
        expectRow(actual[step], expected[step], 0, {});
        expectRow(actual[step], expected[step], 1,
                  step % 3 == 0 ? std::unordered_set<int>{2, 30, 40} : std::unordered_set<int>{2, 10, 20, 30});
        expectRow(actual[step], expected[step], 2, {2, 10, 20, 40});
        expectRow(actual[step], expected[step], 3, {});
        expectRow(actual[step], expected[step], 4, {});
    }
}

TEST_P(UnlistedTreeLogitsProcessorTest, SparseRejectsNonUnitTokenStride) {
    if (!GetParam()) {
        GTEST_SKIP() << "Sparse layout validation";
    }
    TreeLogitsProcessor processor({info(), info()});
    SamplerInputs inputs;
    inputs.logits = torch::ones({64, 2}, torch::TensorOptions().device(torch::kCUDA)).transpose(0, 1);
    auto before = inputs.logits.clone();
    EXPECT_TRUE(processor.process(inputs, 0, 2).has_value());
    EXPECT_TRUE(torch::equal(before, inputs.logits));
}

TEST_P(UnlistedTreeLogitsProcessorTest, DenseSparseRandomLogitsAreExactlyEqual) {
    for (auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        auto original = torch::randn({8, 64}, torch::TensorOptions().device(torch::kCUDA).dtype(dtype));
        original[0][10] = -INFINITY;
        original[0][9] = -INFINITY;
        std::vector<torch::Tensor> outputs;
        for (bool sparse : {false, true}) {
            load("true", sparse);
            auto finished = info("1_20_40");
            finished.dfa_ptr->next(2);
            auto error = info();
            error.dfa_ptr->next(30);
            TreeLogitsProcessor processor({info(), info("1_10"), info("1_20"), info("1_10_30"),
                                           info("missing"), finished, error, info("1", false)});
            SamplerInputs inputs;
            inputs.logits = original.clone();
            ASSERT_FALSE(processor.process(inputs, 0, 8).has_value());
            outputs.push_back(inputs.logits.clone());
        }
        EXPECT_TRUE(torch::equal(outputs[0], outputs[1]));
    }
}

TEST_P(UnlistedTreeLogitsProcessorTest, SparseReloadChangesControlledDomainAndEosZero) {
    if (!GetParam()) {
        GTEST_SKIP() << "GPU cache invalidation";
    }
    TreeLogitsProcessor processor({info()});
    SamplerInputs inputs;
    inputs.logits = torch::ones({1, 64}, torch::TensorOptions().device(torch::kCUDA));
    ASSERT_FALSE(processor.process(inputs, 0, 1).has_value());
    // Remove the old controlled set, and introduce EOS=0 and a vocabulary boundary token.
    std::ofstream file(path_);
    file << R"({"allow_unlisted_tokens":true,"tree_decode_gpu_sparse_mask":true,"start_token_id":1,
        "end_token_id":0,"prefix_dict":{"1":[63],"1_63":[]}})";
    file.close();
    PrefixToCandidateTokens::instance()->reloadPrefixDict(path_);
    inputs.logits.fill_(1);
    auto before = inputs.logits.clone();
    ASSERT_FALSE(processor.process(inputs, 0, 1).has_value());
    expectRow(inputs.logits, before, 0, {0});
    ASSERT_FALSE(processor.updateStatus(tokens({63}, 1), 1).has_value());
    inputs.logits.fill_(1);
    ASSERT_FALSE(processor.process(inputs, 0, 1).has_value());
    expectRow(inputs.logits, before, 0, {63});
}

INSTANTIATE_TEST_SUITE_P(DenseAndSparse, UnlistedTreeLogitsProcessorTest, ::testing::Bool());

#undef EXPECT_SIMILAR

}  // namespace rtp_llm
