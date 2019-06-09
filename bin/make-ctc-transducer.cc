// MIT License
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
#include "fstext/kaldi-fst-io.h"
#include "fstext/fstext-utils.h"

namespace kaldi {

fst::VectorFst<fst::StdArc>* MakeCtcFst(
    const int32 num_symbols, const int32 ctc_symbol) {
  using Weight = fst::StdArc::Weight;
  KALDI_ASSERT(ctc_symbol > 0);
  fst::VectorFst<fst::StdArc>* ofst = new fst::VectorFst<fst::StdArc>();
  for (int32 symbol = 1; symbol <= num_symbols; ++symbol) {
    const auto state = ofst->AddState();
    ofst->AddArc(state, fst::StdArc(symbol, 0, Weight::One(), state));
    ofst->SetFinal(state, Weight::One());
    if (symbol == ctc_symbol) {
      ofst->SetStart(state);
    }
    for (int32 symbol2 = 1; symbol2 < symbol; ++symbol2) {
      const auto state2 = symbol2 - 1;
      ofst->AddArc(
          state,
          fst::StdArc(symbol2,
                      symbol2 == ctc_symbol ? 0 : symbol2,
                      Weight::One(),
                      state2));
      ofst->AddArc(
          state2,
          fst::StdArc(symbol,
                      symbol == ctc_symbol ? 0 : symbol,
                      Weight::One(),
                      state));
    }
  }
  return ofst;
}
}  // namespace kaldi

int main(int argc, char** argv) {
  try {
    using namespace kaldi;

    const char* usage =
        "Creates a transducer to decode the output of a network trained with "
        "CTC.\n"
        "\n"
        "Usage: make-ctc-transducer num-symbols blank-symbol [output-fst]\n"
        " e.g.: make-ctc-transducer 64 1 ctc.fst\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    // Parse arguments.
    const std::string num_symbols_str = po.GetArg(1);
    const std::string ctc_symbol_str = po.GetArg(2);
    const std::string output_fst_filename = po.GetOptArg(3);
    int32 num_symbols = 0;
    ConvertStringToInteger(num_symbols_str, &num_symbols);
    int32 ctc_symbol = 0;
    ConvertStringToInteger(ctc_symbol_str, &ctc_symbol);

    fst::VectorFst<fst::StdArc>* ctc_fst = MakeCtcFst(num_symbols, ctc_symbol);

#if _MSC_VER
    if (output_fst_filename == "")
      _setmode(_fileno(stdout),  _O_BINARY);
#endif

    if (! ctc_fst->Write(output_fst_filename) )
      KALDI_ERR << "make-ctc-transducer: error writing FST to "
                << (output_fst_filename == "" ?
                    "standard output" : output_fst_filename);

    delete ctc_fst;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << e.what();
    return 1;
  }
}