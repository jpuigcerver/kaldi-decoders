SUBDIRS = gmmbin

.PHONY: all install clean distclean $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

install: $(SUBDIRS)
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x install; done

clean:
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x clean; done

distclean:
	-for x in $(SUBDIRS) $(EXT_SUBDIRS); do $(MAKE) -C $$x distclean; done
