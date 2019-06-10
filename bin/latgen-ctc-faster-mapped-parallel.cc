// bin/latgen-ctc-faster-mapped-parallel.cc
//
// Copyright (c) 2018 Joan Puigcerver <joapuipe@gmail.com>
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
#include "util/kaldi-thread.h"

namespace kaldi {
struct CtcLatticeFasterDecoderConfig {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat lattice_beam;
  int32 prune_interval;
  bool determinize; // not inspected by this class... used in
  // command-line program.
  BaseFloat beam_delta; // has nothing to do with beam_ratio
  BaseFloat hash_ratio;
  BaseFloat prune_scale;   // Note: we don't make this configurable on the command line,
  // it's not a very important parameter.  It affects the
  // algorithm that prunes the tokens as we go.
  // Most of the options inside det_opts are not actually queried by the
  // LatticeFasterDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeFaster.
  // delta: a small offset used to measure equality of weights.
  float delta;
  // max_mem: if > 0, determinization will fail and return false when the
  // algorithm's (approximate) memory consumption crosses this threshold.
  int max_mem;
  // minimize: if true, push and minimize after determinization.
  bool minimize;

  CtcLatticeFasterDecoderConfig()
      : beam(16.0),
        max_active(std::numeric_limits<int32>::max()),
        min_active(200),
        lattice_beam(10.0),
        prune_interval(25),
        determinize(true),
        beam_delta(0.5),
        hash_ratio(2.0),
        prune_scale(0.1),
        delta(fst::kDelta),
        max_mem(50000000),
        minimize(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
        "more accurate");
    opts->Register("min-active", &min_active, "Decoder minimum #active states.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam.  Larger->slower, "
        "and deeper lattices");
    opts->Register("prune-interval", &prune_interval, "Interval (in frames) at "
        "which to prune tokens");
    opts->Register("determinize", &determinize, "If true, "
        "determinize the lattice (lattice-determinization, keeping only "
        "best pdf-sequence for each word-sequence).");
    opts->Register("beam-delta", &beam_delta, "Increment used in decoding-- this "
        "parameter is obscure and relates to a speedup in the way the "
        "max-active constraint is applied.  Larger is more accurate.");
    opts->Register("hash-ratio", &hash_ratio, "Setting used in decoder to "
        "control hash behavior");
    // Originally in DeterminizeLatticePhonePrunedOptions.
    opts->Register("delta", &delta, "Tolerance used in determinization");
    opts->Register("max-mem", &max_mem, "Maximum approximate memory usage in "
        "determinization (real usage might be many times this).");
    opts->Register("minimize", &minimize, "If true, push and minimize after "
        "determinization.");
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && max_active > 1 && lattice_beam > 0.0
                     && min_active <= max_active
                     && prune_interval > 0 && beam_delta > 0.0 && hash_ratio >= 1.0
                     && prune_scale > 0.0 && prune_scale < 1.0);
  }
};

class DecodeUtteranceCtcLatticeFasterClass {
 public:

  DecodeUtteranceCtcLatticeFasterClass(
      const Matrix<BaseFloat>* loglikes,
      // Next are part of DecodeUtteranceLatticeFasterClass
      LatticeFasterDecoder * 	decoder,
      const fst::SymbolTable * 	word_syms,
      std::string 	utt,
      BaseFloat 	acoustic_scale,
      bool 	determinize,
      bool 	allow_partial,
      Int32VectorWriter * 	alignments_writer,
      Int32VectorWriter * 	words_writer,
      CompactLatticeWriter * 	compact_lattice_writer,
      LatticeWriter * 	lattice_writer,
      double * 	like_sum,
      int64 * 	frame_sum,
      int32 * 	num_done,
      int32 * 	num_err,
      int32 * 	num_partial) : loglikes_(loglikes), task_(nullptr) {
    task_ = new DecodeUtteranceLatticeFasterClass(
        decoder,
        new kaldi::DecodableMatrixScaled(*loglikes_, acoustic_scale),
        unused_trans_model_,
        word_syms,
        utt,
        acoustic_scale,
        determinize,
        allow_partial,
        alignments_writer,
        words_writer,
        compact_lattice_writer,
        lattice_writer,
        like_sum,
        frame_sum,
        num_done,
        num_err,
        num_partial);
  }

  ~DecodeUtteranceCtcLatticeFasterClass() {
    delete loglikes_;
    delete task_;
  }

  void operator () () {
    (*task_)();
  }

 private:
  const Matrix<BaseFloat> *loglikes_;
  DecodeUtteranceLatticeFasterClass *task_;
  TransitionModel unused_trans_model_;  // Not used.
};
}  // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices, reading log-likelihoods as matrices, using multiple decoding threads\n"
        "Usage: latgen-faster-mapped-parallel [options] (fst-in|fsts-rspecifier) loglikes-rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 1.0;
    CtcLatticeFasterDecoderConfig config;
    TaskSequencerConfig sequencer_config; // has --num-threads option

    std::string word_syms_filename;
    config.Register(&po);
    sequencer_config.Register(&po);

    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        fst_in_str = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        lattice_wspecifier = po.GetArg(3),
        words_wspecifier = po.GetOptArg(4),
        alignment_wspecifier = po.GetOptArg(5);

    // Copy from CtcLatticeFasterDecoderConfig to LatticeFasterDecoderConfig.
    LatticeFasterDecoderConfig lattice_faster_decoder_config;
    lattice_faster_decoder_config.beam = config.beam;
    lattice_faster_decoder_config.max_active = config.max_active;
    lattice_faster_decoder_config.min_active = config.min_active;
    lattice_faster_decoder_config.lattice_beam = config.lattice_beam;
    lattice_faster_decoder_config.prune_interval = config.prune_interval;
    lattice_faster_decoder_config.determinize_lattice = config.determinize;
    lattice_faster_decoder_config.beam_delta = config.beam_delta;
    lattice_faster_decoder_config.hash_ratio = config.hash_ratio;
    lattice_faster_decoder_config.prune_scale = config.prune_scale;
    lattice_faster_decoder_config.det_opts.delta = config.delta;
    lattice_faster_decoder_config.det_opts.max_mem = config.max_mem;
    lattice_faster_decoder_config.det_opts.phone_determinize = false;
    lattice_faster_decoder_config.det_opts.word_determinize = config.determinize;
    lattice_faster_decoder_config.det_opts.minimize = config.minimize;

    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (config.determinize ? compact_lattice_writer.Open(lattice_wspecifier)
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
    Fst<StdArc> *decode_fst = NULL; // only used if there is a single
    // decoding graph.

    TaskSequencer<DecodeUtteranceCtcLatticeFasterClass> sequencer(sequencer_config);
    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);

      {
        for (; !loglike_reader.Done(); loglike_reader.Next()) {
          std::string utt = loglike_reader.Key();
          Matrix<BaseFloat> *loglikes =
              new Matrix<BaseFloat>(loglike_reader.Value());
          loglike_reader.FreeCurrent();
          if (loglikes->NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            delete loglikes;
            continue;
          }

          LatticeFasterDecoder *decoder = new LatticeFasterDecoder(
              *decode_fst, lattice_faster_decoder_config);
          DecodeUtteranceCtcLatticeFasterClass *task =
              new DecodeUtteranceCtcLatticeFasterClass(
                  loglikes, decoder, word_syms, utt,
                  acoustic_scale, config.determinize, allow_partial,
                  &alignment_writer,  &words_writer, &compact_lattice_writer,
                  &lattice_writer, &tot_like, &frame_count, &num_success,
                  &num_fail, NULL);

          sequencer.Run(task); // takes ownership of "task",
          // and will delete it when done.
        }
      }
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!loglike_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no loglikes available.";
          num_fail++;
          continue;
        }
        const Matrix<BaseFloat> *loglikes =
            new Matrix<BaseFloat>(loglike_reader.Value(utt));
        if (loglikes->NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_fail++;
          delete loglikes;
          continue;
        }
        fst::VectorFst<StdArc> *fst =
            new fst::VectorFst<StdArc>(fst_reader.Value());
        LatticeFasterDecoder *decoder =
            new LatticeFasterDecoder(lattice_faster_decoder_config, fst);
        DecodeUtteranceCtcLatticeFasterClass *task =
            new DecodeUtteranceCtcLatticeFasterClass(
                loglikes, decoder, word_syms, utt, acoustic_scale,
                config.determinize, allow_partial, &alignment_writer,
                &words_writer, &compact_lattice_writer, &lattice_writer,
                &tot_like, &frame_count, &num_success, &num_fail, NULL);
        sequencer.Run(task); // takes ownership of "task",
        // and will delete it when done.
      }
    }
    sequencer.Wait();

    delete decode_fst;

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Decoded with " << sequencer_config.num_threads << " threads.";
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor per thread assuming 100 frames/sec is "
              << (sequencer_config.num_threads*elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
