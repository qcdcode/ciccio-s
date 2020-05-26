include_HEADERS=ciccio-s.hpp

#add ciccio-s library
lib_LIBRARIES=$(top_builddir)/lib/libciccio-s.a
__top_builddir__lib_libciccio_s_a_SOURCES=

%D%/git_info.hpp: $(top_srcdir)/.git/HEAD $(top_srcdir)/.git/index $(top_srcdir)/config/get_git_info.sh Makefile
	@echo "Reading git version"
	@$(top_srcdir)/config/get_git_info.sh $(top_srcdir) > $@
$(top_srcdir)/.git/HEAD:
$(top_srcdir)/.git/index:


BUILT_SOURCES=%D%/git_info.hpp

include $(top_srcdir)/src/base/Makefile.am
