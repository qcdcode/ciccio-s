#include m4 dir
ACLOCAL_AMFLAGS=-I m4

#init to empty
include_HEADERS=
bin_PROGRAMS=
BUILT_SOURCES=
CLEANFILES=

#rule to create assembly output and get it filtered
%.s: %.cpp %.o
	$(CXXCOMPILE) -S $< -c -o -|$(top_srcdir)/config/asm_inspect.sh > $@

include $(top_srcdir)/projects/Makefile.am
include $(top_srcdir)/src/Makefile.am

