# kaldi-decoders

[![Build Status](https://travis-ci.org/jpuigcerver/kaldi-decoders.svg?branch=ctc-decoder)](https://travis-ci.org/jpuigcerver/kaldi-decoders)

This repository contains a set of custom Kaldi decoders.

The directory has a similar structure of Kaldi's src dir: i.e. gmmbin contains
decoders using Gaussian Mixture Models emissions.


## Compile & install

You need to define the environment variable KALDI_ROOT to point to your Kaldi distribution.

```bash
export KALDI_ROOT=/path/to/your/kaldi/distribution
make depend
make
```

Once compiled, you can install the binary to PREFIX/bin (by default
PREFIX=/usr/local):

```bash
make install
```

## Available decoders

### LazyLM decoders

Kaldi decoders typically recive the HCLG finite state transducer to perform
the decoding. However, when a large vocabulary is used, or even more, when
a large n-gram language model is used for G, the amount of memory required
to store the HCLG transducer may be too large.

These decoders receive the HCL and G transducers separately, and then
make a dynamic (on-the-fly) composition to obtain the HCLG transducer while
decoding. The dynamic composition, together with beam pruning, makes the
amount of required memory much smaller.

- **decode-lazylm-faster-mapped**: Decode utterances, reading
   log-likelihoods as matrices.
- **gmm-decode-lazylm-faster**: Decode features using GMM-based model.
- **gmm-latgen-lazylm-faster**: Generate lattices using GMM-based model.
- **latgen-lazylm-faster-mapped**: Generate lattices, reading
   log-likelihoods as matrices.
