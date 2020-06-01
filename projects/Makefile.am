AUTOMAKE_OPTIONS=subdir-objects

LDADD=$(top_builddir)/lib/libciccio-s.a
AM_CPPFLAGS=-I$(top_srcdir)/src

bin_PROGRAMS+= \
        $(top_builddir)/bin/main

__top_builddir__bin_main_SOURCES=%D%/main.cpp

if GENERATE_SIDE_ASSEMBLY

BUILT_SOURCES+=%D%/main.s

endif
