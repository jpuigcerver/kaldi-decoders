// bin/decode-lazylm-faster-mapped.cc
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
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc

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
        "Decode, reading log-likelihoods as matrices and doing the composition "
        "of HCL and G on-the-fly. You can use this decoder to perform decoding "
        "using large grammars (i.e. large n-gram LMs).\n"
        "\n"
        "Usage:   decode-lazylm-faster-mapped [options] <model-in> "
        "<hcl-in> <g-in> <loglikes-rspecifier> <words-wspecifier> "
        "[<alignments-wspecifier>]\n";
    ParseOptions po(usage);
    bool binary = true;
    BaseFloat acoustic_scale = 0.1;
    bool allow_partial = true;
    std::string word_syms_filename;
    FasterDecoderOptions decoder_opts;
    fst::CacheOptions cache_config;
    int gc_limit = 536870912;  // 512MB
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("allow-partial", &allow_partial, "Produce output even when final state was not reached");
    po.Register("compose-gc", &cache_config.gc,
                "If false, any expanded state during the composition will be "
                "cached. If true, the cache will be garbage collected when it "
                "grows past --compose-gc-limit bytes.");
    po.Register("compose-gc-limit", &gc_limit,
                "Number of bytes allowed in the composition cache before "
                "garbage collection.");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Read(argc, argv);
    cache_config.gc_limit = gc_limit;

    if (po.NumArgs() < 5 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        hcl_in_filename = po.GetArg(2),
        g_in_filename = po.GetArg(3),
        loglikes_rspecifier = po.GetArg(4),
        words_wspecifier = po.GetArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
      word_syms = fst::SymbolTable::ReadText(word_syms_filename);
      if (!word_syms)
        KALDI_ERR << "Could not read symbol table from file "<<word_syms_filename;
    }

    SequentialBaseFloatMatrixReader loglikes_reader(loglikes_rspecifier);

    // It's important that we initialize decode_fst after loglikes_reader, as it
    // can prevent crashes on systems installed without enough virtual memory.
    // It has to do with what happens on UNIX systems if you call fork() on a
    // large process: the page-table entries are duplicated, which requires a
    // lot of virtual memory.
    VectorFst<StdArc> *hcl_fst = fst::ReadFstKaldi(hcl_in_filename);
    VectorFst<StdArc> *g_fst = fst::ReadFstKaldi(g_in_filename);

    // On-demand composition of HCL and G
    fst::ComposeFst<StdArc> decode_fst = fst::TableComposeFst(
        *hcl_fst, *g_fst, cache_config);

    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    FasterDecoder decoder(decode_fst, decoder_opts);

    Timer timer;

    for (; !loglikes_reader.Done(); loglikes_reader.Next()) {
      std::string key = loglikes_reader.Key();
      const Matrix<BaseFloat> &loglikes (loglikes_reader.Value());

      if (loglikes.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key;
        num_fail++;
        continue;
      }

      DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);
      decoder.Decode(&decodable);

      VectorFst<LatticeArc> decoded;  // linear FST.

      if ( (allow_partial || decoder.ReachedFinal())
           && decoder.GetBestPath(&decoded) ) {
        num_success++;
        if (!decoder.ReachedFinal())
          KALDI_WARN << "Decoder did not reach end-state, outputting partial traceback.";

        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        frame_count += loglikes.NumRows();

        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

        words_writer.Write(key, words);
        if (alignment_writer.IsOpen())
          alignment_writer.Write(key, alignment);
        if (word_syms != NULL) {
          std::cerr << key << ' ';
          for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }
        BaseFloat like = -weight.Value1() -weight.Value2();
        tot_like += like;
        KALDI_LOG << "Log-like per frame for utterance " << key << " is "
                  << (like / loglikes.NumRows()) << " over "
                  << loglikes.NumRows() << " frames.";

      } else {
        num_fail++;
        KALDI_WARN << "Did not successfully decode utterance " << key
                   << ", len = " << loglikes.NumRows();
      }
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";

    delete word_syms;
    delete hcl_fst;
    delete g_fst;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
