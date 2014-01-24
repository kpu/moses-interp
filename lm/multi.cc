#include "lm/multi.hh"

#include "lm/enumerate_vocab.hh"
#include "lm/model.hh"
#include "util/exception.hh"
#include "util/murmur_hash.hh"

#include <boost/unordered_map.hpp>

#include <assert.h>
#include <math.h>
#include <stdint.h>

namespace lm { namespace multi {

void Vocabulary::Configure(WordIndex begin_sentence, WordIndex end_sentence) {
  SetSpecial(begin_sentence, end_sentence, 0);
}

WordIndex Vocabulary::Index(const StringPiece &str) const {
  UTIL_THROW(util::Exception, "Vocabulary index lookup is not implemented.");
}

class EnumerateWrap : public EnumerateVocab {
  public:
    EnumerateWrap(std::vector<WordIndex> &build, std::size_t models, EnumerateVocab *notify)
      : mapping_(build), current_model_(0), models_(models), next_(0), notify_(notify) {}

    void Add(WordIndex index, const StringPiece &str) {
      uint64_t hash = util::MurmurHashNative(str.data(), str.size());
      std::pair<boost::unordered_map<uint64_t, WordIndex>::iterator, bool> res(string_to_index_.insert(std::make_pair(hash, next_)));
      WordIndex id;
      if (res.second) {
        id = next_++;
        if (notify_) notify_->Add(id, str);
        if ((id + 1) * models_ > mapping_.size()) {
          mapping_.resize((id + 1) * models_);
        }
      } else {
        id = res.first->second;
      }
      mapping_[id * models_ + current_model_] = index;
    }

    void NextModel() {
      ++current_model_;
      assert(current_model_ <= models_);
    }

    WordIndex Lookup(const StringPiece &str) {
      boost::unordered_map<uint64_t, WordIndex>::const_iterator i = string_to_index_.find(util::MurmurHashNative(str.data(), str.size()));
      return i == string_to_index_.end() ? 0 : i->second;
    }

  private:
    std::vector<WordIndex> &mapping_;
    std::size_t current_model_;
    const std::size_t models_;

    boost::unordered_map<uint64_t, WordIndex> string_to_index_;

    WordIndex next_;

    EnumerateVocab *const notify_;
};

const char kMagicLine[] = "Interpolated LM specification";

Model::Model(const char *config_file, ngram::Config config) {
  util::FilePiece file(config_file);
  UTIL_THROW_IF(kMagicLine != file.ReadLine(), util::Exception, "First line of " << config_file << " was not \"" << kMagicLine << "\"");
  std::vector<std::string> files;
  while (true) {
    try {
      weights_.push_back(file.ReadFloat());
      assert(!isnan(weights_.back()));
      UTIL_THROW_IF(weights_.back() < 0.0, util::Exception, "Negative weight " << weights_.back());
    } catch (const util::EndOfFileException &e) { break; } 
    file.SkipSpaces();
    files.resize(files.size() + 1);
    StringPiece line = file.ReadLine();
    files.back().assign(line.data(), line.size());
  }
  EnumerateWrap wrap(mapping_, weights_.size(), config.enumerate_vocab);
  config.enumerate_vocab = &wrap;

  std::size_t total_state = 0;
  unsigned char order = 0;
  for (std::size_t i = 0; i < weights_.size(); ++i, wrap.NextModel()) {
    ngram::ModelType model_type = ngram::PROBING;
    const char *name = files[i].c_str();
    RecognizeBinary(name, model_type);
    switch (model_type) {
      case ngram::PROBING:
        backends_.push_back(new ngram::ProbingModel(name, config));
        break;
      case ngram::REST_PROBING:
        backends_.push_back(new ngram::RestProbingModel(name, config));
        break;
      case ngram::TRIE:
        backends_.push_back(new ngram::TrieModel(name, config));
        break;
      case ngram::QUANT_TRIE:
        backends_.push_back(new ngram::QuantTrieModel(name, config));
        break;
      case ngram::ARRAY_TRIE:
        backends_.push_back(new ngram::ArrayTrieModel(name, config));
        break;
      case ngram::QUANT_ARRAY_TRIE:
        backends_.push_back(new ngram::QuantArrayTrieModel(name, config));
        break;
      default:
        UTIL_THROW(util::Exception, "Unrecognized kenlm model type " << model_type);
    }
    total_state += backends_.back().StateSize();
    order = std::max(order, backends_.back().Order());
  }
  fake_vocab_.Configure(wrap.Lookup("<s>"), wrap.Lookup("</s>"));
  State begin_sentence, null_context;
  begin_sentence.bytes.resize(total_state);
  null_context.bytes.resize(total_state);
  char *begin_base = &begin_sentence.bytes[0];
  char *null_base = &null_context.bytes[0];
  for (std::size_t i = 0; i < backends_.size(); ++i) {
    backends_[i].BeginSentenceWrite(begin_base);
    backends_[i].NullContextWrite(null_base);
    begin_base += backends_[i].StateSize();
    null_base += backends_[i].StateSize();
  }
  Init(begin_sentence, null_context, fake_vocab_, order);
}

class Mix {
  public:
    Mix() : accum_(0.0) {}

    void Add(float weight, float value) {
      assert(!isnan(value));
      accum_ += weight * powf(10.0, value);
      assert(!isnan(accum_));
    }

    float Finish() {
      float ret = log10(accum_);
      assert(!isnan(ret));
      return ret;
    }

  private:
    float accum_;
};

FullScoreReturn Model::FullScore(const State &in_state, const WordIndex new_word, State &out_state) const {
  Mix mix;
  const char *in = &in_state.bytes[0];
  out_state.bytes.resize(BeginSentenceState().bytes.size());
  char *out = &out_state.bytes[0];
  for (std::size_t i = 0; i < backends_.size(); ++i) {
    mix.Add(weights_[i], backends_[i].Score(in, new_word, out));
    in += backends_[i].StateSize();
    out += backends_[i].StateSize();
  }
  FullScoreReturn ret;
  ret.prob = mix.Finish();
  ret.rest = ret.prob;
  // Other fields undefined...
  assert(!isnan(ret.prob));
  return ret;
}

FullScoreReturn Model::FullScoreForgotState(const WordIndex *context_rbegin, const WordIndex *context_rend, const WordIndex new_word, State &out_state) const {
  Mix mix;
  out_state.bytes.resize(BeginSentenceState().bytes.size());
  char *out = &out_state.bytes[0];
  for (std::size_t i = 0; i < backends_.size(); ++i) {
    mix.Add(weights_[i], backends_[i].FullScoreForgotState(context_rbegin, context_rend, new_word, out).prob);
    out += backends_[i].StateSize();
  }
  FullScoreReturn ret;
  ret.prob = mix.Finish();
  ret.rest = ret.prob;
  assert(!isnan(ret.prob));
  return ret;
}

void Model::GetState(const WordIndex *context_rbegin, const WordIndex *context_rend, State &out_state) const {
  out_state.bytes.resize(BeginSentenceState().bytes.size());
  char *out = &out_state.bytes[0];
  for (std::size_t i = 0; i < backends_.size(); ++i) {
    backends_[i].GetState(context_rbegin, context_rend, out);
    out += backends_[i].StateSize();
  }
}

FullScoreReturn Model::ExtendLeft(
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
        unsigned char &next_use) const {
  UTIL_THROW(util::Exception, "ExtendLeft is Unimplemented");
}

float Model::UnRest(const uint64_t *pointers_begin, const uint64_t *pointers_end, unsigned char first_length) const {
  UTIL_THROW(util::Exception, "UnRest is Unimplemented");
}

} } // namespaces
