#ifndef LM_MULTI__
#define LM_MULTI__

#include "lm/facade.hh"
#include "lm/state.hh"
#include "lm/virtual_interface.hh"

#include <boost/ptr_container/ptr_vector.hpp>

#include <vector>

namespace lm {

namespace ngram { class Config; }

namespace multi {

extern const char kMagicLine[];

struct State {
  std::vector<char> bytes;
  int Compare(const State &other) const {
    // HACK
    assert(bytes.size() == other.bytes.size());
    assert(bytes.size() % sizeof(ngram::State) == 0);
    const char *a = &bytes[0], *b = &other.bytes[0];
    for (; a != &bytes[0] + bytes.size(); a += sizeof(ngram::State), b += sizeof(ngram::State)) {
      int ret = reinterpret_cast<const ngram::State*>(a)->Compare(*reinterpret_cast<const ngram::State*>(b));
      if (ret) return ret;
    }
    return 0;
  }
};

// This is a fake vocabulary for now.  It will only work with BeginSentence and EndSentence.
class Vocabulary : public base::Vocabulary {
  public:
    Vocabulary() {}

    void Configure(WordIndex begin_sentence, WordIndex end_sentence);

    // Throws an exception.
    WordIndex Index(const StringPiece &str) const;
};

class Model : public base::ModelFacade<Model, State, Vocabulary> {
  public:
    Model(const char *config_file, ngram::Config config);

    FullScoreReturn FullScore(const State &in_state, const WordIndex new_word, State &out_state) const;

    FullScoreReturn FullScoreForgotState(const WordIndex *context_rbegin, const WordIndex *context_rend, const WordIndex new_word, State &out_state) const;

    void GetState(const WordIndex *context_rbegin, const WordIndex *context_rend, State &out_state) const;

    // Unimplemented
    FullScoreReturn ExtendLeft(
        // Additional context in reverse order.  This will update add_rend to 
        const WordIndex *add_rbegin, const WordIndex *add_rend,
        // Backoff weights to use.  
        const float *backoff_in,
        // extend_left returned by a previous query.
        uint64_t extend_pointer,
        // Length of n-gram that the pointer corresponds to.  
        unsigned char extend_length,
        // Where to write additional backoffs for [extend_length + 1, min(Order() - 1, return.ngram_length)]
        float *backoff_out,
        // Amount of additional content that should be considered by the next call.
        unsigned char &next_use) const;

    // Unimplemented
    float UnRest(const uint64_t *pointers_begin, const uint64_t *pointers_end, unsigned char first_length) const;

  private:
    boost::ptr_vector<base::Model> backends_;

    // 2d array mapping vocab ids, flattend to 1d.  
    std::vector<WordIndex> mapping_;

    Vocabulary fake_vocab_;

    std::vector<float> weights_;
};

} // namespace multi
} // namespace lm

#endif // LM_MULTI__
