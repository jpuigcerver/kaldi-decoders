ifndef KALDI_ROOT
  $(error KALDI_ROOT environment variable is undefined)
endif



EXTRA_CXXFLAGS = -Wno-sign-compare -Wno-unused-variable -I$(KALDI_ROOT)/src
include $(KALDI_ROOT)/src/kaldi.mk

BINFILES = decode-lazylm-faster-mapped latgen-lazylm-faster-mapped

OBJFILES =

TESTFILES =

ADDLIBS = \
	$(KALDI_ROOT)/src/decoder/kaldi-decoder.a \
	$(KALDI_ROOT)/src/lat/kaldi-lat.a \
	$(KALDI_ROOT)/src/fstext/kaldi-fstext.a \
        $(KALDI_ROOT)/src/hmm/kaldi-hmm.a \
	$(KALDI_ROOT)/src/tree/kaldi-tree.a \
        $(KALDI_ROOT)/src/util/kaldi-util.a \
	$(KALDI_ROOT)/src/matrix/kaldi-matrix.a \
	$(KALDI_ROOT)/src/base/kaldi-base.a



.PHONY: all install

all: $(BINFILES)

PREFIX=/usr/local
install: $(BINFILES)
	test -d $(PREFIX) || mkdir -p $(PREFIX)/bin
	install -m 0755 $(BINFILES) $(PREFIX)/bin

include $(KALDI_ROOT)/src/makefiles/default_rules.mk
