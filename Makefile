SUBDIRS = bin gmmbin

.PHONY: all install clean distclean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

install: $(SUBDIRS)
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x install; done

clean:
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x clean; done

depend:
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x depend; done

distclean:
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x distclean; done
