#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"

using namespace std;

namespace rtp_llm {

TreeLogitsProcessor::TreeLogitsProcessor(std::vector<StreamTreeInfo> tree_infos): tree_infos_(tree_infos) {}

std::optional<ErrorInfo>
TreeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    auto batch_size = size();
    RTP_LLM_CHECK(batch_size == finish_idx - start_idx);
    bool                             need_process = false;
    std::vector<std::vector<size_t>> batch_candidate_token_ids(batch_size);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) {
            continue;
        }
        const auto& candidate_token_ids = info.dfa_ptr->getCandidateTokenIds();
        batch_candidate_token_ids[i]    = candidate_token_ids;
        if (candidate_token_ids.size() > 0) {
            need_process = true;
        }
    }
    // If no beams need processing, return early
    if (!need_process) {
        return std::nullopt;
    }

    auto   batch_logits     = inputs.logits.narrow(0, start_idx, batch_size);
    size_t vocab_size       = batch_logits.size(1);
    torch::Tensor batch_vocab_mask;
    const auto prefix = PrefixToCandidateTokens::instance();
    if (prefix->allowUnlistedTokens()) {
        // Validate before indexing; an out-of-vocabulary controlled token is a
        // configuration error, not permission to fall back to unconstrained decoding.
        for (auto token : prefix->controlledTokens()) {
            if (token < 0 || static_cast<size_t>(token) >= vocab_size) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "tree decode controlled token is outside the vocabulary");
            }
        }
        auto mask_cpu = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size}, torch::kUInt8);
        auto mask = mask_cpu.data_ptr<uint8_t>();
        for (size_t i = 0; i < batch_size; ++i) {
            if (!tree_infos_[i].in_tree_mode) {
                continue;
            }
            for (auto token : prefix->controlledTokens()) {
                mask[i * vocab_size + token] = 1;
            }
            for (auto token : batch_candidate_token_ids[i]) {
                if (token >= vocab_size) {
                    return ErrorInfo(ErrorCode::INVALID_PARAMS, "tree decode candidate is outside the vocabulary");
                }
                mask[i * vocab_size + token] = 0;
            }
        }
        batch_vocab_mask = mask_cpu.to(batch_logits.device());
    } else {
        batch_vocab_mask = generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
    }
    maskLogits(batch_logits, batch_vocab_mask);
    return std::nullopt;
}

void TreeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamTreeInfo> new_tree_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_tree_infos.push_back(tree_infos_[src_batch_idx].copy());
    }
    tree_infos_ = std::move(new_tree_infos);
}

std::optional<ErrorInfo> TreeLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    RTP_LLM_CHECK(size() == (size_t)new_tokens.size(0));

    for (size_t i = 0; i < size(); i++) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode)
            continue;

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens.size(1));
        }

        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = new_tokens.data_ptr<int>()[i * new_tokens.size(1) + j + offset];
            info.dfa_ptr->next(current_token_id);
            if (info.dfa_ptr->hasError()) {
                break;
            }
        }

        info.current_output_length += num_new_tokens;
    }
    return std::nullopt;
}

TreeLogitsProcessorPtr TreeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                              int32_t                        num) {
    if (!PrefixToCandidateTokens::instance()->initSuccess()) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<TreeLogitsProcessor>();
    for (size_t i = 0; i < num; i++) {
        StreamTreeInfo              tree_info(PrefixToCandidateTokens::instance()->initSuccess(),
                                 generate_input->inputLength(),
                                 0,
                                 generate_input->generate_config->hasNumBeams()
                                     || generate_input->generate_config->num_return_sequences > 1,
                                 std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()));
        std::vector<StreamTreeInfo> tree_infos       = {tree_info};
        auto                        single_processor = std::make_shared<TreeLogitsProcessor>(tree_infos);

        processor_ptr->insert(single_processor, 1);
    }

    return processor_ptr;
}

std::vector<std::string> TreeLogitsProcessor::getStatus() {
    std::vector<std::string> status_list;
    for (const auto& tree_info : tree_infos_) {
        status_list.push_back(tree_info.dfa_ptr->status());
    }
    return status_list;
}

}  // namespace rtp_llm
