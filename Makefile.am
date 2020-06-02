#include m4 dir
ACLOCAL_AMFLAGS=-I m4

#init to empty
include_HEADERS=
bin_PROGRAMS=
BUILT_SOURCES=
CLEANFILES=
generate_assembly=

#rule to create assembly output and get it filtered
%.s: %.cpp %.o
	$(CXXCOMPILE) -S $< -c -o -|$(top_srcdir)/config/asm_inspect.sh > $@

include $(top_srcdir)/src/Makefile.am
include $(top_srcdir)/projects/Makefile.am

#generate assembly if needed
if GENERATE_SIDE_ASSEMBLY
BUILT_SOURCES+=$(generate_assembly)
CLEANFILES+=$(generate_assembly)
endif

