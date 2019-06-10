#!/bin/bash
set -e;

MAXPAR=4;
[ -z "$KALDI_ROOT" ] && KALDI_ROOT=/tmp/kaldi;

if [ ! -d "$KALDI_ROOT" ]; then
  git clone https://github.com/kaldi-asr/kaldi.git "${KALDI_ROOT}";
fi;
cd "${KALDI_ROOT}";

cd tools;
# Install OpenFST.
make -j$MAXPAR openfst \
     CC="$CC" CXX="$CXX" CXXFLAGS="-g" \
     OPENFST_CONFIGURE="--disable-static --enable-shared --disable-bin --disable-dependency-tracking";
cd ..;

cd src;
touch .short_version   # Make version short, or else ccache will miss everything.
CC="$CC" CXX="$CXX" CXXFLAGS="-g" \
  ./configure --use-cuda=no --mathlib=ATLAS;
make -j$MAXPAR clean;
make -j$MAXPAR depend CC="$CC" CXX="$CXX";

for dir in base matrix util tree hmm gmm transform feat fstext lat decoder; do
  make -j$MAXPAR -C "$dir" CC="$CC" CXX="$CXX";
done
