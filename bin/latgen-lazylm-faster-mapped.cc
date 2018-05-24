// bin/latgen-lazylm-faster-mapped.cc
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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"

namespace fst {

template<class Arc>
ComposeFst<Arc> TableComposeFst(
    const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
    const CacheOptions& cache_opts = CacheOptions()) {
  typedef Fst<Arc> F;
  typedef SortedMatcher<F> SM;
  typedef TableMatcher<F> TM;
  typedef ArcLookAheadMatcher<SM> LA_SM;
  typedef SequenceComposeFilter<TM, LA_SM> SCF;
  typedef LookAheadComposeFilter<SCF, TM, LA_SM, MATCH_INPUT> LCF;
  typedef PushWeightsComposeFilter<LCF, TM, LA_SM, MATCH_INPUT> PWCF;
  typedef PushLabelsComposeFilter<PWCF, TM, LA_SM, MATCH_INPUT> PWLCF;
  TM* lam1 = new TM(ifst1, MATCH_OUTPUT);
  LA_SM* lam2 = new LA_SM(ifst2, MATCH_INPUT);
  PWLCF* laf = new PWLCF(ifst1, ifst2, lam1, lam2);
  ComposeFstImplOptions<TM, LA_SM, PWLCF> opts(cache_opts, lam1, lam2, laf);
  return ComposeFst<Arc>(ifst1, ifst2, opts);
}

}  // namespace fst


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices, reading log-likelihoods as matrices and doing the "
        "composition of HCL and G on-the-fly. You can use this decoder to "
        "perform decoding using large grammars (i.e. large n-gram LMs).\n"
        "\n"
        "Usage:   latgen-lazylm-faster-mapped [options] <model-in> "
        "<hcl-in> <g-in> <loglikes-rspecifier> <lattice-wspecifier> "
        "[<words-wspecifier> [<alignments-wspecifier>]]\n";

    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    std::string retry_beam_str = "";
    LatticeFasterDecoderConfig config;
    std::string word_syms_filename;
    fst::CacheOptions cache_config;
    int gc_limit = 536870912;  // 512MB

    config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoos.");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");
    po.Register("compose-gc", &cache_config.gc,
                "If false, any expanded state during the composition will be "
                "cached. If true, the cache will be garbage collected when it "
                "grows past --compose-gc-limit bytes.");
    po.Register("compose-gc-limit", &gc_limit,
                "Number of bytes allowed in the composition cache before "
                "garbage collection.");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output].");
    po.Register("retry-beam", &retry_beam_str,
                "Space-separated list of beam to try when decoding fails");
    po.Read(argc, argv);
    cache_config.gc_limit = gc_limit;

    if (po.NumArgs() < 5 || po.NumArgs() > 7) {
      po.PrintUsage();
      exit(1);
    }

    std::vector<BaseFloat> retry_beam, retry_beam_tmp;
    kaldi::SplitStringToFloats(retry_beam_str, " ", true, &retry_beam_tmp);
    for (auto b : retry_beam_tmp) {
      if (b < config.beam) {
        KALDI_WARN << "Ignoring retry beam " << b
                   << ", which is lower than the default beam " << config.beam;
      } else {
        retry_beam.push_back(b);
      }
    }
    retry_beam.push_back(config.beam);
    std::sort(retry_beam.begin(), retry_beam.end());

    std::string model_in_filename = po.GetArg(1),
        hcl_in_str = po.GetArg(2),
        g_in_str = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        lattice_wspecifier = po.GetArg(5),
        words_wspecifier = po.GetOptArg(6),
        alignment_wspecifier = po.GetOptArg(7);

    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    const bool is_table_hcl =
        ClassifyRspecifier(hcl_in_str, NULL, NULL) != kNoRspecifier;
    const bool is_table_g =
        ClassifyRspecifier(g_in_str, NULL, NULL) != kNoRspecifier;

    if (!is_table_hcl && !is_table_g) {
      SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      // It's important that we initialize decode_fst after loglikes_reader,
      // as it can prevent crashes on systems installed without enough virtual
      // memory.
      // It has to do with what happens on UNIX systems if you call fork() on a
      // large process: the page-table entries are duplicated, which requires a
      // lot of virtual memory.
      VectorFst<StdArc> *hcl_fst = fst::ReadFstKaldi(hcl_in_str);
      VectorFst<StdArc> *g_fst = fst::ReadFstKaldi(g_in_str);

      // On-demand composition of HCL and G
      fst::ComposeFst<StdArc> decode_fst = fst::TableComposeFst(
          *hcl_fst, *g_fst, cache_config);
      timer.Reset();

      {
        LatticeFasterDecoder decoder(decode_fst, config);

        for (; !loglike_reader.Done(); loglike_reader.Next()) {
          std::string utt = loglike_reader.Key();
          Matrix<BaseFloat> loglikes (loglike_reader.Value());
          loglike_reader.FreeCurrent();
          if (loglikes.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }

          DecodableMatrixScaledMapped decodable(trans_model, loglikes,
                                                acoustic_scale);
          double like = -std::numeric_limits<double>::infinity();
          bool success = false;
          for (const auto& beam : retry_beam) {
            auto retry_config = config;
            retry_config.beam = beam;
            decoder.SetOptions(retry_config);
            if (DecodeUtteranceLatticeFaster(
                    decoder, decodable, trans_model, word_syms, utt,
                    acoustic_scale, determinize, allow_partial,
                    &alignment_writer, &words_writer,
                    &compact_lattice_writer, &lattice_writer, &like)) {
              tot_like += like;
              frame_count += loglikes.NumRows();
              num_success++;
              success = true;
              break;
            }
          }
          if (!success) { num_fail++; }
        }
      }
      // delete these only after decoder goes out of scope.
      delete hcl_fst;
      delete g_fst;
    } else if(is_table_hcl && is_table_g) {
      // We have different FSTs for different utterances.
      SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      RandomAccessTableReader<fst::VectorFstHolder> hcl_reader(hcl_in_str);
      RandomAccessTableReader<fst::VectorFstHolder> g_reader(g_in_str);
      for (; !loglike_reader.Done(); loglike_reader.Next()) {
        std::string utt = loglike_reader.Key();
        const Matrix<BaseFloat> &loglikes = loglike_reader.Value();
        loglike_reader.FreeCurrent();
        if (loglikes.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_fail++;
          continue;
        }
        if (!hcl_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no HCL is available.";
          num_fail++;
          continue;
        }
        if (!g_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no G is available.";
          num_fail++;
          continue;
        }

        // On-demand composition of HCL and G
        fst::ComposeFst<StdArc> decode_fst = fst::TableComposeFst(
            hcl_reader.Value(utt), g_reader.Value(utt), cache_config);

        LatticeFasterDecoder decoder(decode_fst, config);
        DecodableMatrixScaledMapped decodable(trans_model, loglikes,
                                              acoustic_scale);
        double like = -std::numeric_limits<double>::infinity();
        bool success = false;
        for (const auto& beam : retry_beam) {
          auto retry_config = config;
          retry_config.beam = beam;
          decoder.SetOptions(retry_config);
          if (DecodeUtteranceLatticeFaster(
                  decoder, decodable, trans_model, word_syms, utt,
                  acoustic_scale, determinize, allow_partial,
                  &alignment_writer, &words_writer,
                  &compact_lattice_writer, &lattice_writer, &like)) {
            tot_like += like;
            frame_count += loglikes.NumRows();
            num_success++;
            success = true;
            break;
          }
        }
        if (!success) { num_fail++; }
      }
    } else {
      KALDI_ERR << "The decoding of tables/non-tables and match-type that you "
                << "supplied is not currently supported. Either implement "
                << "this, ask the maintainers to implement it, or call this "
                << "program differently.";
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like/frame_count) << " over " << frame_count<<" frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }
}
