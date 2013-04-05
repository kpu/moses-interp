// $Id$

/***********************************************************************
Moses - factored phrase-based language decoder
Copyright (C) 2006 University of Edinburgh

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
***********************************************************************/

#include <boost/lexical_cast.hpp>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include "lm/binary_format.hh"
#include "lm/enumerate_vocab.hh"
#include "lm/left.hh"
#include "lm/model.hh"
#include "util/tokenize_piece.hh"

#include "Ken.h"
#include "Base.h"
#include "moses/FFState.h"
#include "moses/TypeDef.h"
#include "moses/Util.h"
#include "moses/FactorCollection.h"
#include "moses/Phrase.h"
#include "moses/InputFileStream.h"
#include "moses/StaticData.h"
#include "moses/ChartHypothesis.h"
#include "moses/Incremental.h"

#include <boost/shared_ptr.hpp>

using namespace std;

namespace Moses {
namespace {

template <class S> struct KenLMState : public FFState {
  S state;
  int Compare(const FFState &o) const {
    const KenLMState<S> &other = static_cast<const KenLMState<S> &>(o);
    return other.state.Compare(state);
  }
};

class MappingBuilder : public lm::EnumerateVocab {
  public:
    MappingBuilder(FactorCollection &factorCollection, std::vector<lm::WordIndex> &mapping)
      : m_factorCollection(factorCollection), m_mapping(mapping) {}

    void Add(lm::WordIndex index, const StringPiece &str) {
      std::size_t factorId = m_factorCollection.AddFactor(str)->GetId();
      if (m_mapping.size() <= factorId) {
        // 0 is <unk> :-)
        m_mapping.resize(factorId + 1);
      }
      m_mapping[factorId] = index;
    }

  private:
    FactorCollection &m_factorCollection;
    std::vector<lm::WordIndex> &m_mapping;
};

template <class Model> class VocabWrap {
  public:
    typedef typename Model::State State;

    VocabWrap(const std::string &file, FactorType factor_type, bool lazy) 
      : builder_(FactorCollection::Instance(), m_lmIdLookup),
        model_(file.c_str(), MakeConfig(lazy)),
        factor_type_(factor_type) {}

    const State &BeginSentenceState() const {
      return model_.BeginSentenceState();
    }

    const State &NullContextState() const {
      return model_.NullContextState();
    }

    float Score(const State &state, const Word &word, State &out) const {
      return model_.Score(state, TranslateID(word), out);
    }

    float EndSentence(const Hypothesis &hypo, State &out) const {
      lm::WordIndex indices[KENLM_MAX_ORDER - 1];
      return model_.FullScoreForgotState(indices, LastIDs(hypo, indices), model_.GetVocabulary().EndSentence(), out).prob;
    }

    void GetState(const Hypothesis &hypo, State &out) const {
      lm::WordIndex indices[KENLM_MAX_ORDER - 1];
      model_.GetState(indices, LastIDs(hypo, indices), out);
    }

    unsigned char Order() const {
      return model_.Order();
    }
 
  private:
    lm::ngram::Config MakeConfig(bool lazy) {
      lm::ngram::Config config;
      IFVERBOSE(1) {
        config.messages = &std::cerr;
      } else {
        config.messages = NULL;
      }
      config.enumerate_vocab = &builder_;
      config.load_method = lazy ? util::LAZY : util::POPULATE_OR_READ;
      return config;
    }

    lm::WordIndex TranslateID(const Word &word) const {
      std::size_t factor = word.GetFactor(factor_type_)->GetId();
      return (factor >= m_lmIdLookup.size() ? 0 : m_lmIdLookup[factor]);
    }

    // Convert last words of hypothesis into vocab ids, returning an end pointer.  
    lm::WordIndex *LastIDs(const Hypothesis &hypo, lm::WordIndex *indices) const {
      lm::WordIndex *index = indices;
      lm::WordIndex *end = indices + model_.Order() - 1;
      int position = hypo.GetCurrTargetWordsRange().GetEndPos();
      for (; ; ++index, --position) {
        if (index == end) return index;
        if (position == -1) {
          *index = model_.GetVocabulary().BeginSentence();
          return index + 1;
        }
        *index = TranslateID(hypo.GetWord(position));
      }
    }

    std::vector<lm::WordIndex> m_lmIdLookup;

    MappingBuilder builder_;

    Model model_;

    FactorType factor_type_;
};

class InterpWrap {
  public:
    struct State {
      lm::ngram::State first, second;
      int Compare(const State &other) const {
        int ret = first.Compare(other.first);
        if (ret) return ret;
        return second.Compare(other.second);
      }
    };

    InterpWrap(const std::string &file, FactorType factorType, bool lazy) 
      : first_(FirstName(file), factorType, lazy), second_(SecondName(file), factorType, lazy) {
      begin_sentence_.first = first_.BeginSentenceState();
      begin_sentence_.second = second_.BeginSentenceState();
      null_context_.first = first_.NullContextState();
      null_context_.second = second_.NullContextState();
      util::TokenIter<util::SingleCharacter> it(file, ':');
      ++it;
      ++it;
      first_weight_ = boost::lexical_cast<float>(*it);
      second_weight_ = 1.0 - first_weight_;
    }

    const State &BeginSentenceState() const {
      return begin_sentence_;
    }

    const State &NullContextState() const {
      return null_context_;
    }

    float Score(const State &state, const Word &word, State &out) const {
      return Mix(
          first_.Score(state.first, word, out.first),
          second_.Score(state.second, word, out.second));
    }

    float EndSentence(const Hypothesis &hypo, State &out) const {
      return Mix(
          first_.EndSentence(hypo, out.first),
          second_.EndSentence(hypo, out.second));
    }

    void GetState(const Hypothesis &hypo, State &out) const {
      first_.GetState(hypo, out.first);
      second_.GetState(hypo, out.second);
    }

    unsigned char Order() const {
      return 5;
    }

  private:
    float Mix(float first, float second) const {
      return log10(pow(10.0, first) * first_weight_ + pow(10.0, second) * second_weight_);
    }

    static std::string FirstName(const std::string &file) {
      return util::TokenIter<util::SingleCharacter>(file, ':')->as_string();
    }
    static std::string SecondName(const std::string &file) {
      util::TokenIter<util::SingleCharacter> it(file, ':');
      ++it;
      return it->as_string();
    }

    const VocabWrap<lm::ngram::Model> first_;
    const VocabWrap<lm::ngram::QuantArrayTrieModel> second_;

    State begin_sentence_, null_context_;

    float first_weight_, second_weight_;
};

/*
 * An implementation of single factor LM using Ken's code.
 */
template <class Model> class LanguageModelKen : public LanguageModel {
  public:
    LanguageModelKen(const std::string &file, FactorType factorType, bool lazy);

    LanguageModel *Duplicate() const;

    bool Useable(const Phrase &phrase) const {
      return (phrase.GetSize()>0 && phrase.GetFactor(0, m_factorType) != NULL);
    }

    std::string GetScoreProducerDescription(unsigned) const {
      std::ostringstream oss;
      oss << "LM_" << (unsigned)m_ngram->Order() << "gram";
      return oss.str();
    }

    const FFState *EmptyHypothesisState(const InputType &/*input*/) const {
      KenLMState<typename Model::State> *ret = new KenLMState<typename Model::State>();
      ret->state = m_ngram->BeginSentenceState();
      return ret;
    }

    void CalcScore(const Phrase &phrase, float &fullScore, float &ngramScore, size_t &oovCount) const;

    FFState *Evaluate(const Hypothesis &hypo, const FFState *ps, ScoreComponentCollection *out) const;

    FFState *EvaluateChart(const ChartHypothesis& cur_hypo, int featureID, ScoreComponentCollection *accumulator) const;

    void IncrementalCallback(Incremental::Manager &manager) const {
      //manager.LMCallback(*m_ngram, m_lmIdLookup);
    }

  private:
    LanguageModelKen(const LanguageModelKen<Model> &copy_from);

    boost::shared_ptr<Model> m_ngram;

    FactorType m_factorType;
    
    const Factor *m_beginSentenceFactor;
};

template <class Model> LanguageModelKen<Model>::LanguageModelKen(const std::string &file, FactorType factorType, bool lazy)
  : m_ngram(new Model(file, factorType, lazy)),
    m_factorType(factorType),
    m_beginSentenceFactor(FactorCollection::Instance().AddFactor(BOS_)) {}

template <class Model> LanguageModel *LanguageModelKen<Model>::Duplicate() const {
  return new LanguageModelKen<Model>(*this);
}

template <class Model> LanguageModelKen<Model>::LanguageModelKen(const LanguageModelKen<Model> &copy_from) :
    m_ngram(copy_from.m_ngram),
    m_factorType(copy_from.m_factorType),
    m_beginSentenceFactor(copy_from.m_beginSentenceFactor) {}

template <class Model> void LanguageModelKen<Model>::CalcScore(const Phrase &phrase, float &fullScore, float &ngramScore, size_t &oovCount) const {
  fullScore = 0;
  ngramScore = 0;
  oovCount = 0;

  if (!phrase.GetSize()) return;

  typename Model::State state;
  size_t position;
  if (m_beginSentenceFactor == phrase.GetWord(0).GetFactor(m_factorType)) {
    state = m_ngram->BeginSentenceState();
    position = 1;
  } else {
    state = m_ngram->NullContextState();
    position = 0;
  }
  
  size_t ngramBoundary = m_ngram->Order() - 1;
  size_t end_loop = std::min(ngramBoundary, phrase.GetSize());
  for (; position < end_loop; ++position) {
    typename Model::State out_state;
    fullScore += m_ngram->Score(state, phrase.GetWord(position), out_state);
    state = out_state;
  }
  float before_boundary = fullScore;
  for (; position < phrase.GetSize(); ++position) {
    typename Model::State out_state;
    fullScore += m_ngram->Score(state, phrase.GetWord(position), out_state);
    state = out_state;
  }
  ngramScore = TransformLMScore(fullScore - before_boundary);
  fullScore = TransformLMScore(fullScore);
}

template <class Model> FFState *LanguageModelKen<Model>::Evaluate(const Hypothesis &hypo, const FFState *ps, ScoreComponentCollection *out) const {
  typedef KenLMState<typename Model::State> StateWrap;
  const typename Model::State &in_state = static_cast<const StateWrap&>(*ps).state;

  std::auto_ptr<StateWrap> ret(new StateWrap());
  
  if (!hypo.GetCurrTargetLength()) {
    ret->state = in_state;
    return ret.release();
  }

  const std::size_t begin = hypo.GetCurrTargetWordsRange().GetStartPos();
  //[begin, end) in STL-like fashion.
  const std::size_t end = hypo.GetCurrTargetWordsRange().GetEndPos() + 1;
  const std::size_t adjust_end = std::min(end, begin + m_ngram->Order() - 1);

  std::size_t position = begin;
  typename Model::State aux_state;
  typename Model::State *state0 = &ret->state, *state1 = &aux_state;

  float score = m_ngram->Score(in_state, hypo.GetWord(position), *state0);
  ++position;
  for (; position < adjust_end; ++position) {
    score += m_ngram->Score(*state0, hypo.GetWord(position), *state1);
    std::swap(state0, state1);
  }

  if (hypo.IsSourceCompleted()) {
    // Score end of sentence.  
    score += m_ngram->EndSentence(hypo, ret->state);
  } else if (adjust_end < end) {
    // Get state after adding a long phrase.  
    /*std::vector<lm::WordIndex> indices(m_ngram->Order() - 1);
    const lm::WordIndex *last = LastIDs(hypo, &indices.front());*/
    m_ngram->GetState(hypo, ret->state);
  } else if (state0 != &ret->state) {
    // Short enough phrase that we can just reuse the state.  
    ret->state = *state0;
  }

  score = TransformLMScore(score);

  if (OOVFeatureEnabled()) {
    std::vector<float> scores(2);
    scores[0] = score;
    scores[1] = 0.0;
    out->PlusEquals(this, scores);
  } else {
    out->PlusEquals(this, score);
  }

  return ret.release();
}

template <class Model> FFState *LanguageModelKen<Model>::EvaluateChart(const ChartHypothesis& hypo, int featureID, ScoreComponentCollection *accumulator) const {
  return NULL;
}

} // namespace

LanguageModel *ConstructKenLM(const std::string &file, FactorType factorType, bool lazy) {
  try {
    lm::ngram::ModelType model_type;
    if (lm::ngram::RecognizeBinary(file.c_str(), model_type)) {
      switch(model_type) {
        case lm::ngram::PROBING:
          return new LanguageModelKen<VocabWrap<lm::ngram::ProbingModel> >(file,  factorType, lazy);
        case lm::ngram::REST_PROBING:
          return new LanguageModelKen<VocabWrap<lm::ngram::RestProbingModel> >(file, factorType, lazy);
        case lm::ngram::TRIE:
          return new LanguageModelKen<VocabWrap<lm::ngram::TrieModel> >(file, factorType, lazy);
        case lm::ngram::QUANT_TRIE:
          return new LanguageModelKen<VocabWrap<lm::ngram::QuantTrieModel> >(file, factorType, lazy);
        case lm::ngram::ARRAY_TRIE:
          return new LanguageModelKen<VocabWrap<lm::ngram::ArrayTrieModel> >(file, factorType, lazy);
        case lm::ngram::QUANT_ARRAY_TRIE:
          return new LanguageModelKen<VocabWrap<lm::ngram::QuantArrayTrieModel> >(file, factorType, lazy);
        default:
          std::cerr << "Unrecognized kenlm model type " << model_type << std::endl;
          abort();
      }
    } else {
      return new LanguageModelKen<VocabWrap<lm::ngram::ProbingModel> >(file, factorType, lazy);
    }
  } catch (util::ErrnoException &e) {
    return new LanguageModelKen<InterpWrap>(file, factorType, lazy);
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    abort();
  }
}

}

