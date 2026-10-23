#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/models/logits_processor/DFAUtil.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"

#include <chrono>
#include <fstream>
#include <memory>
#include <thread>
#include <unistd.h>

using namespace std;

namespace rtp_llm {

class DFAUtilTest: public ::testing::Test {
protected:
};

TEST_F(DFAUtilTest, testSimple) {
    std::vector<int>              re = {3};
    StringContainDFA<size_t, int> dfa(re);

    ASSERT_FALSE(dfa.isFinished());
    ASSERT_EQ(0, dfa.next(1));
    ASSERT_EQ(0, dfa.next(2));
    ASSERT_EQ(1, dfa.next(3));
    ASSERT_EQ(1, dfa.next(1));
    ASSERT_EQ(1, dfa.next(1));
    ASSERT_EQ(1, dfa.next(3));
    ASSERT_TRUE(dfa.isFinished());
}

TEST_F(DFAUtilTest, testComplex) {
    std::vector<int>              re = {1, 1, 2};
    StringContainDFA<size_t, int> dfa(re);

    ASSERT_FALSE(dfa.isFinished());
    ASSERT_EQ(1, dfa.next(1));
    ASSERT_EQ(2, dfa.next(1));
    ASSERT_EQ(2, dfa.next(1));
    ASSERT_EQ(0, dfa.next(3));
    ASSERT_EQ(1, dfa.next(1));
    ASSERT_EQ(2, dfa.next(1));
    ASSERT_EQ(3, dfa.next(2));
    ASSERT_TRUE(dfa.isFinished());
}

class TreeDFATest: public ::testing::Test {
protected:
    void SetUp() override {
        std::string   json = R"({
            "start_token_id": 100, "end_token_id": 999,
            "prefix_dict": {
                "100":     [10, 20],
                "100_10":  [30, 40],
                "100_20":  [50],
                "100_10_30": [999]
            }
        })";
        std::string   path = "/tmp/test_tree_dfa.json";
        std::ofstream f(path);
        f << json;
        f.close();
        prefix_ = PrefixToCandidateTokens::instance();
        prefix_->reloadPrefixDict(path);
    }

    std::shared_ptr<PrefixToCandidateTokens> prefix_;
};

TEST_F(TreeDFATest, InvalidTokenEntersErrorState) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    std::string s = dfa.next(10);
    ASSERT_EQ("100_10", s);
    ASSERT_FALSE(dfa.hasError());

    // Invalid transition: "100_10" -> input 99 → error state
    s = dfa.next(99);
    ASSERT_TRUE(dfa.hasError());
    // status_ stays at last valid state
    ASSERT_EQ("100_10", s);
}

TEST_F(TreeDFATest, ErrorStateOnlyAllowsEOS) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    dfa.next(10);
    dfa.next(99);  // invalid → error
    ASSERT_TRUE(dfa.hasError());

    auto candidates = dfa.getCandidateTokenIds();
    ASSERT_EQ(1u, candidates.size());
    ASSERT_EQ(prefix_->endTokenId(), candidates[0]);
}

TEST_F(TreeDFATest, ErrorStateFreezesFurtherInput) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    dfa.next(10);
    std::string before_error = dfa.status();
    dfa.next(99);  // invalid → error
    ASSERT_TRUE(dfa.hasError());

    // Further inputs are ignored, status doesn't change
    std::string s = dfa.next(20);
    ASSERT_EQ(before_error, s);
    ASSERT_TRUE(dfa.hasError());
}

TEST_F(TreeDFATest, ValidPathToEnd) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    ASSERT_FALSE(dfa.isFinished());
    dfa.next(10);  // "100_10"
    ASSERT_FALSE(dfa.isFinished());
    dfa.next(30);  // "100_10_30"
    ASSERT_FALSE(dfa.isFinished());
    dfa.next(999);  // end token
    ASSERT_TRUE(dfa.isFinished());
}

TEST_F(TreeDFATest, FinishedDFAIgnoresFurtherInput) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    dfa.next(10);
    dfa.next(30);
    dfa.next(999);
    ASSERT_TRUE(dfa.isFinished());

    // Further input on a finished DFA is ignored: the status must stay unchanged.
    std::string finished_status = dfa.status();
    ASSERT_EQ(finished_status, dfa.next(10));
    ASSERT_TRUE(dfa.isFinished());
}

TEST_F(TreeDFATest, GetCandidateTokenIds) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    auto                       candidates = dfa.getCandidateTokenIds();
    std::unordered_set<size_t> candidate_set(candidates.begin(), candidates.end());
    // At start "100", candidates should be {10, 20}
    ASSERT_TRUE(candidate_set.count(10));
    ASSERT_TRUE(candidate_set.count(20));

    dfa.next(10);  // "100_10"
    candidates    = dfa.getCandidateTokenIds();
    candidate_set = std::unordered_set<size_t>(candidates.begin(), candidates.end());
    // At "100_10", candidates should be {30, 40}
    ASSERT_TRUE(candidate_set.count(30));
    ASSERT_TRUE(candidate_set.count(40));
}

TEST_F(TreeDFATest, ForceSetStatus) {
    TreeDFA<std::string, size_t> dfa(prefix_);

    dfa.forceSetStatus("100_20");
    ASSERT_EQ("100_20", dfa.status());

    auto                       candidates = dfa.getCandidateTokenIds();
    std::unordered_set<size_t> candidate_set(candidates.begin(), candidates.end());
    ASSERT_TRUE(candidate_set.count(50));
}

// Runtime coverage for the negative-candidate fix: negative ids are skipped at load, so
// getCandidateTokenIds() must not hit its negative-id throw path during logits processing.
TEST_F(TreeDFATest, NegativeCandidateTokensSkippedAtRuntime) {
    std::string   json = R"({
        "start_token_id": 100, "end_token_id": 999,
        "prefix_dict": {"100": [-5, 10, -1, 20]}
    })";
    std::string   path = "/tmp/test_tree_dfa_neg.json";
    std::ofstream f(path);
    f << json;
    f.close();

    auto prefix = PrefixToCandidateTokens::instance();
    prefix->reloadPrefixDict(path);
    ASSERT_TRUE(prefix->initSuccess());

    TreeDFA<std::string, size_t> dfa(prefix);

    std::vector<size_t> candidates;
    ASSERT_NO_THROW(candidates = dfa.getCandidateTokenIds());

    std::unordered_set<size_t> candidate_set(candidates.begin(), candidates.end());
    ASSERT_TRUE(candidate_set.count(10));
    ASSERT_TRUE(candidate_set.count(20));
    ASSERT_EQ(2u, candidate_set.size()) << "negative candidate ids must not survive to runtime";
}


class UnlistedTreeDFATest: public ::testing::Test {
protected:
    void SetUp() override {
        path_ = "/tmp/unlisted_tree_dfa_" + std::to_string(getpid()) + ".json";
        load(2);
    }
    void TearDown() override { std::remove(path_.c_str()); }
    void load(int end) {
        std::ofstream file(path_);
        file << R"({"allow_unlisted_tokens":true,"start_token_id":1,"end_token_id":)" << end
             << R"(,"prefix_dict":{"1":[10,20],"1_10":[30],"1_10_30":[],"1_20":[]}})";
        file.close();
        prefix_ = PrefixToCandidateTokens::instance();
        prefix_->reloadPrefixDict(path_);
        ASSERT_TRUE(prefix_->initSuccess());
    }
    std::string path_;
    PrefixToCandidateTokensPtr prefix_;
};

TEST_F(UnlistedTreeDFATest, OrdinaryTokensResetToRoot) {
    TreeDFA<std::string, int> dfa(prefix_);
    EXPECT_EQ("1", dfa.next(99));
    EXPECT_EQ("1_10", dfa.next(10));
    EXPECT_EQ("1", dfa.next(99));
    EXPECT_EQ("1", dfa.next(98));
    const auto candidates = dfa.getCandidateTokenIds();
    EXPECT_EQ((std::unordered_set<size_t>{10, 20}),
              std::unordered_set<size_t>(candidates.begin(), candidates.end()));
    EXPECT_EQ("1_20", dfa.next(20));
    EXPECT_FALSE(dfa.hasError());
}

TEST_F(UnlistedTreeDFATest, ResetDoesNotAllowOldContinuation) {
    TreeDFA<std::string, int> dfa(prefix_);
    dfa.next(10);
    dfa.next(99);
    EXPECT_EQ("1", dfa.next(30));
    EXPECT_TRUE(dfa.hasError());
    EXPECT_EQ((std::vector<size_t>{2}), dfa.getCandidateTokenIds());
}

TEST_F(UnlistedTreeDFATest, EndTokenRemainsControlledAndLeafCanRestart) {
    TreeDFA<std::string, int> dfa(prefix_);
    dfa.next(2); // EOS is not a root candidate, even though absent from all values.
    EXPECT_TRUE(dfa.hasError());
    EXPECT_FALSE(dfa.isFinished());
    dfa.next(99);
    EXPECT_FALSE(dfa.hasError());
    dfa.next(20);
    EXPECT_EQ((std::vector<size_t>{2}), dfa.getCandidateTokenIds());
    dfa.next(99); // A leaf does not force EOS in the new mode.
    EXPECT_EQ("1", dfa.status());
    dfa.next(20);
    dfa.next(2);
    ASSERT_TRUE(dfa.isFinished());
    EXPECT_EQ((std::vector<size_t>{2}), dfa.getCandidateTokenIds());
    dfa.next(99); // Also reset if the stream did not terminate on configured E.
    EXPECT_FALSE(dfa.isFinished());
    EXPECT_EQ("1", dfa.status());
}

TEST_F(UnlistedTreeDFATest, ZeroEndTokenAndMissingState) {
    load(0);
    TreeDFA<std::string, int> dfa(prefix_);
    EXPECT_FALSE(dfa.isFinished());
    dfa.forceSetStatus("missing");
    EXPECT_EQ((std::vector<size_t>{0}), dfa.getCandidateTokenIds());
    EXPECT_EQ("1", dfa.next(99));
    dfa.next(20);
    dfa.next(0);
    EXPECT_TRUE(dfa.isFinished());
}

TEST_F(UnlistedTreeDFATest, CopiesMaintainIndependentState) {
    TreeDFA<std::string, size_t> parent(prefix_);
    parent.next(10);
    TreeDFA<std::string, size_t> first(parent), second(parent);
    first.next(99);
    second.next(30);
    EXPECT_EQ("1_10", parent.status());
    EXPECT_EQ("1", first.status());
    EXPECT_EQ("1_10_30", second.status());
}

TEST_F(UnlistedTreeDFATest, InvalidNumericInputIsNotAnOrdinaryToken) {
    TreeDFA<std::string, int> signed_dfa(prefix_);
    signed_dfa.next(-1);
    EXPECT_TRUE(signed_dfa.hasError());
    TreeDFA<std::string, size_t> unsigned_dfa(prefix_);
    unsigned_dfa.next(std::numeric_limits<size_t>::max());
    EXPECT_TRUE(unsigned_dfa.hasError());
}

}  // namespace rtp_llm
