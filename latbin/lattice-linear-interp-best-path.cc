// latbin/lattice-linear-interp-best-path
//
// Copyright (c) 2016 Joan Puigcerver <joapuipe@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

namespace fst {

void ConvertLatticeWeight(const kaldi::CompactLatticeWeight& iw,
                          LogArc::Weight* ow) {
  KALDI_ASSERT(ow != NULL);
  *ow = LogArc::Weight(iw.Weight().Value1() + iw.Weight().Value2());
}

}  // namespace fst

namespace kaldi {

// This removes the total weight from a CompactLattice. Since the total backward
// score is in log likelihood domain, and the lattice weights are in negative
// log likelihood domain, the total weight is *added* to the weight of the final
// states. This is equivalent to dividing the probability of each path by the
// total probability over all paths. There is an additional weight to control
// the relative contribution of individual lattices-- the log of the weight will
// become the total weight of the lattice.
bool CompactLatticeNormalize(CompactLattice *clat, BaseFloat weight) {
  if (weight <= 0.0) {
    KALDI_WARN << "Weights must be positive; found: " << weight;
    return false;
  }

  if (clat->Properties(fst::kTopSorted, false) == 0) {
    if (fst::TopSort(clat) == false) {
      KALDI_WARN << "Cycles detected in lattice: cannot normalize.";
      return false;
    }
  }

  vector<double> beta;
  if (!ComputeCompactLatticeBetas(*clat, &beta)) {
    KALDI_WARN << "Failed to compute backward probabilities on lattice.";
    return false;
  }

  typedef CompactLattice::Arc::StateId StateId;
  StateId start = clat->Start();  // Should be 0
  BaseFloat total_backward_cost = beta[start];

  total_backward_cost -= Log(weight);

  for (fst::StateIterator<CompactLattice> sit(*clat); !sit.Done(); sit.Next()) {
    CompactLatticeWeight f = clat->Final(sit.Value());
    LatticeWeight w = f.Weight();
    w.SetValue1(w.Value1() + total_backward_cost);
    f.SetWeight(w);
    clat->SetFinal(sit.Value(), f);
  }
  return true;
}

// This is a wrapper for SplitStringToFloats, with added checks to make sure
// the weights are valid probabilities.
void SplitStringToWeights(const string &full, const char *delim,
                          vector<BaseFloat> *out) {
  vector<BaseFloat> tmp;
  SplitStringToFloats(full, delim, true /*omit empty strings*/, &tmp);
  if (tmp.size() != out->size()) {
    KALDI_WARN << "Expecting " << out->size() << " weights, found " << tmp.size()
               << ": using uniform weights.";
    return;
  }
  BaseFloat sum = 0;
  for (vector<BaseFloat>::const_iterator itr = tmp.begin();
       itr != tmp.end(); ++itr) {
    if (*itr < 0.0) {
      KALDI_WARN << "Cannot use negative weight: " << *itr << "; input string: "
                 << full << "\n\tUsing uniform weights.";
      return;
    }
    sum += (*itr);
  }
  if (sum != 1.0) {
    KALDI_WARN << "Weights sum to " << sum << " instead of 1: renormalizing";
    for (vector<BaseFloat>::iterator itr = tmp.begin();
         itr != tmp.end(); ++itr)
      (*itr) /= sum;
  }
  out->swap(tmp);
}

}  // end namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "lattice-interp-best-path <lattice-rspecifier1> <lattice-rspecifier2> "
        "[<lattice-rspecifier3> ...] <transcriptions-wspecifier>"
        "\n"
        " e.g.: lattice-interp-best-path --acoustic-scales=0.1:0.1 "
        "--lat-weights=0.3:0.7 ark:1a.lats ark:1b.lats ark:1.tra ark:1.ali";

    ParseOptions po(usage);
    std::string acoustic_scales_str = "";
    std::string lm_scales_str = "";
    std::string lat_weights_str = "";
    std::string insertion_penalties_str = "";
    std::string alignments_wspecifier = "";

    po.Register("acoustic-scales", &acoustic_scales_str,
                "Colon-separated list of scaling factor for acoustic "
                "likelihoods. If empty, all lattice use 1.0.");
    po.Register("insertion-penalties", &insertion_penalties_str,
                "Colon-separated list of (word) insertion penalties. If empty, "
                "all lattice use 0.0.");
    po.Register("lat-weights", &lat_weights_str,
                "Colon-separated list of weights for each rspecifier (which "
                "should be non-negative and sum to 1). If empty, the weights"
                "are distributed uniformly.");
    po.Register("lm-scales", &lm_scales_str,
                "Colon-separated list of scaling factor for LM probabilities."
                "If empty, all lattices use 1.0.");
    po.Register("write-best-alignments", &alignments_wspecifier,
                "If given, the best alignments for the transcription will be "
                "written to this table.");
    po.Read(argc, argv);

    const int num_args = po.NumArgs();
    if (num_args < 3) {
      po.PrintUsage();
      exit(1);
    }

    const std::string lats_rspecifier1 = po.GetArg(1),
        transcriptions_wspecifier = po.GetArg(num_args);

    SequentialCompactLatticeReader clat_reader1(lats_rspecifier1);
    std::vector<RandomAccessCompactLatticeReader*> clat_reader_vec(
        num_args-2, static_cast<RandomAccessCompactLatticeReader*>(NULL));
    std::vector<string> clat_rspec_vec(num_args-2);
    for (int i = 2; i < num_args; ++i) {
      clat_reader_vec[i-2] = new RandomAccessCompactLatticeReader(po.GetArg(i));
      clat_rspec_vec[i-2] = po.GetArg(i);
    }
    Int32VectorWriter transcriptions_writer(transcriptions_wspecifier);
    Int32VectorWriter alignments_writer(alignments_wspecifier);

    std::vector<BaseFloat> lat_weights(num_args - 1, 1.0/(num_args - 1));
    if (!lat_weights_str.empty())
      SplitStringToWeights(lat_weights_str, ":", &lat_weights);

    std::vector<BaseFloat> acoustic_scales(num_args - 1, 1.0);
    SplitStringToFloats(acoustic_scales_str, ":", true, &acoustic_scales);
    if (acoustic_scales.size() != num_args - 1)
      KALDI_WARN << "Expecting " << num_args - 1 << " acoustic scales, found "
                 << acoustic_scales.size() << ": using 1.0 acoustic scales.";

    std::vector<BaseFloat> lm_scales(num_args - 1, 1.0);
    SplitStringToFloats(lm_scales_str, ":", true, &lm_scales);
    if (lm_scales.size() != num_args - 1)
      KALDI_WARN << "Expecting " << num_args - 1 << " lm scales, found "
                 << lm_scales.size() << ": using 1.0 lm scales.";

    std::vector<BaseFloat> insertion_penalties(num_args - 1, 1.0);
    SplitStringToFloats(insertion_penalties_str, ":", true,
                        &insertion_penalties);
    if (insertion_penalties.size() != num_args - 1)
      KALDI_WARN << "Expecting " << num_args - 1 << " insertion penalties, "
                 << insertion_penalties_str.size() << " found: "
                 << "using 0.0 insertion penlaties.";


    int32 n_utts = 0, n_total_lats = 0, n_success = 0, n_missing = 0,
        n_other_errors = 0;

    for (; !clat_reader1.Done(); clat_reader1.Next()) {
      std::string key = clat_reader1.Key();
      CompactLattice clat1 = clat_reader1.Value();
      clat_reader1.FreeCurrent();
      n_utts++;
      n_total_lats++;
      fst::ScaleLattice(fst::LatticeScale(lm_scales[0], acoustic_scales[0]),
                        &clat1);
      kaldi::AddWordInsPenToCompactLattice(insertion_penalties[0], &clat1);
      bool success = CompactLatticeNormalize(&clat1, lat_weights[0]);
      if (!success) {
        KALDI_WARN << "Could not normalize lattice for system 1, utterance: "
                   << key;
        n_other_errors++;
        continue;
      }
      fst::VectorFst<fst::LogArc> fst1;
      fst::ConvertLattice(clat1, &fst1);
      clat1.DeleteStates();  // free memory


      for (int32 i = 0; i < num_args-2; ++i) {
        if (clat_reader_vec[i]->HasKey(key)) {
          CompactLattice clat2 = clat_reader_vec[i]->Value(key);
          n_total_lats++;
          fst::ScaleLattice(
              fst::LatticeScale(lm_scales[i + 1], acoustic_scales[i + 1]),
              &clat2);
          kaldi::AddWordInsPenToCompactLattice(insertion_penalties[i + 1],
                                               &clat2);
          success = CompactLatticeNormalize(&clat2, lat_weights[i+1]);
          if (!success) {
            KALDI_WARN << "Could not normalize lattice for system "<< (i + 2)
                       << ", utterance: " << key;
            n_other_errors++;
            continue;
          }
          fst::VectorFst<fst::LogArc> fst2;
          fst::ConvertLattice(clat2, &fst2);
          fst::Union(&fst1, fst2);
        } else {
          KALDI_WARN << "No lattice found for utterance " << key << " for "
                     << "system " << (i + 2) << ", rspecifier: "
                     << clat_rspec_vec[i];
          n_missing++;
        }
      }

      fst::DeterminizeFst<fst::LogArc> det_fst(fst1);
      fst::VectorFst<fst::LogArc> best_path;
      fst::ShortestPath(fst1, &best_path);
      if (best_path.Start() == fst::kNoStateId) {
        KALDI_WARN << "Best-path failed for key " << key;
        n_other_errors++;
      } else {
        std::vector<int32> alignment;
        std::vector<int32> words;
        fst::LogArc::Weight weight;
        GetLinearSymbolSequence(best_path, &alignment, &words, &weight);
        KALDI_LOG << "For utterance " << key << ", best cost " << weight
                  << " over " << alignment.size() << " frames.";
        transcriptions_writer.Write(key, words);
        if (alignments_wspecifier != "")
          alignments_writer.Write(key, alignment);
        n_success++;
        //n_frame += alignment.size();
        //tot_weight = Times(tot_weight, weight);
      }
    }

    DeletePointers(&clat_reader_vec);

    // The success code we choose is that at least one lattice was output,
    // and more lattices were "all there" than had at least one system missing.
    return (n_success != 0 && n_missing < (n_success - n_missing) ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
