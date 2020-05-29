AUTOMAKE_OPTIONS=subdir-objects

LDADD=$(top_builddir)/lib/libciccio-s.a
AM_CPPFLAGS=-I$(top_srcdir)/src

#init to empty
bin_PROGRAMS=
include_HEADERS=
BUILT_SOURCES=
CLEANFILES=

bin_PROGRAMS+= \
        $(top_builddir)/bin/main

__top_builddir__bin_main_SOURCES=%D%/main.cpp
