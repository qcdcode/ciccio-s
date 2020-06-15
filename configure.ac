AC_INIT([ciccio-s],[1.0],[fr.sanfilippo@gmail.com])

#include all configure flags
CONFIGURE_FLAGS="$*"
AC_DEFINE_UNQUOTED([CONFIGURE_FLAGS],["$CONFIGURE_FLAGS"],[Flags passed to configure])

#silent automake
AM_SILENT_RULES([yes])

#get configuration time
AC_MSG_CHECKING([configuration date and time])
CONFIGURE_TIME=`date`
AC_DEFINE_UNQUOTED([CONFIGURE_TIME], "$CONFIGURE_TIME", [time when configured])
AC_MSG_RESULT([$CONFIG_TIME])

#parameters definition
AC_CONFIG_SRCDIR([src])
AC_CONFIG_AUX_DIR([config])
AM_INIT_AUTOMAKE([1.14 no-define foreign subdir-objects])
AM_CONFIG_HEADER(src/config.hpp)
AC_CONFIG_MACRO_DIR([m4])
AC_CANONICAL_HOST

#checks for programs
AC_PROG_CC
AC_PROG_FC
AC_PROG_CXX
AC_PROG_RANLIB
AC_LANG([C++])

LIBRARY_RESULT="---"
SUMMARY_RESULT="---"

AX_ASSEMBLY_REPORT

#cuda
AX_CHECK_CUDA
AM_CONDITIONAL([USE_CUDA],[test "$CXX" == "nvcc"])

#check if we enable fortran
AC_ARG_ENABLE(fortran,
	AS_HELP_STRING([--enable-fortran],[Enable fortran support]),
	enable_fortran="${enableval}",
	enable_fortran="no")
AC_MSG_RESULT([enabling fortran... $enable_fortran])

#split fortran flags into LDFLAGS and LIBS
if test "$enable_fortran" == "yes"
then
	AC_F77_LIBRARY_LDFLAGS
	fortran_ldflags=` echo "$FLIBS" | grep -o -- '\(^\| \)-L\([[^\"[:space:]]]\+\|\"[[^\"[:space:]]]\+\"\)' | tr '\n' ' ' | sed 's/^[[ \t]]*//;s/[[ \t]]*$//' | sed 's/  +/ /g'`
	fortran_libs=`       echo "$FLIBS" | grep -o -- '\(^\| \)-l\([[^\"[:space:]]]\+\|\"[[^\"[:space:]]]\+\"\)' | tr '\n' ' ' | sed 's/^[[ \t]]*//;s/[[ \t]]*$//' | sed 's/  +/ /g'`
	LDFLAGS="$fortran_ldflags $LDFLAGS"
	LIBS="$fortran_libs $LIBS"
	FLIBS=""
fi

#check if we enable MPI autosearch
AC_ARG_ENABLE(MPI-autosearch,
	AS_HELP_STRING([--enable-MPI-autosearch],[Enable MPI autosearch]),
	enable_MPI_autosearch="${enableval}",
	enable_MPI_autosearch="yes")
AC_MSG_RESULT([enabling MPI autosearch... $enable_MPI_autosearch])

#set MPI for fortran
if test "$enable_fortran" == "yes" -a "$enable_MPI_autosearch" == "yes"
then
	AC_LANG_PUSH([Fortran])
	LX_FIND_MPI
	AC_LANG_POP([Fortran])
	AC_MSG_RESULT([with Fortran MPI link options... $MPI_FLDFLAGS])
	LIBS="$MPI_FLIBS $LIBS"
	LDFLAGS="$MPI_FLDFLAGS $LDFLAGS"
fi

#set MPI for c++
if test "$enable_MPI_autosearch" == "yes"
then
	AC_LANG_PUSH([C++])
	LX_FIND_MPI
	AC_LANG_POP([C++])
	AC_MSG_RESULT([with C++ MPI link options... $MPI_CXXLDFLAGS])
fi

#MPI
AC_ARG_ENABLE(MPI,
	AS_HELP_STRING([--disable-MPI],[Enable MPI]),
	enable_MPI="${enableval}",
	enable_MPI="yes")
AM_CONDITIONAL([USE_MPI],[test "$enable_MPI" == "yes" ],[true],[false])
AC_MSG_RESULT([enabling MPI... $enable_MPI])
SUMMARY_RESULT="$SUMMARY_RESULT
MPI enabled         : $enable_MPI"

if test "$enable_MPI" == "yes"
then
	# Check if we have MPI
	if test "$have_CXX_mpi" == "yes" -o "$enable_MPI_autosearch" != "yes"
	then
		CPPFLAGS="$MPI_CXXFLAGS $CPPFLAGS"
		LIBS="$MPI_CXXLIBS $LIBS"
		LDFLAGS="$MPI_CXXLDFLAGS $LDFLAGS"
		AC_DEFINE([USE_MPI],1,[Enable MPI])
	else
		AC_MSG_ERROR(["Unable to find MPI"])
	fi
fi

#check and enable openmp
AX_OPENMP
CFLAGS="$CFLAGS $OPENMP_CFLAGS"
CPPFLAGS="$CPPFLAGS $OPENMP_CPPFLAGS"
CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"

# Threads debug
AC_ARG_ENABLE(threads-debug,
	AS_HELP_STRING([--enable-threads-debug],[Enable threads debug]),
	enable_threads_debug="${enableval}",
	enable_threads_debug="no")
if test "$enable_threads_debug" == "yes"
then
	AC_DEFINE([USE_THREADS_DEBUG],1,"Using threads debug")
fi
AC_MSG_RESULT([enabling threads debug... $enable_threads_debug])
SUMMARY_RESULT="$SUMMARY_RESULT
threads debug       : $enable_threads_debug"

# Search for pthread
AX_SUBPACKAGE(threads,pthread.h,pthread,pthread_getconcurrency,THREADS,autouse)

#AX_CXXFLAGS_WARN_ALL
AC_MSG_CHECKING(whether compiler understands -Wall)
OLD_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS -Wall"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[])],
  AC_MSG_RESULT(yes),
  AC_MSG_RESULT(no)
  CXXFLAGS="$OLD_CXXFLAGS")

AX_SUBPACKAGE(eigen,Eigen/Dense,,,EIGEN,".",".")

#simd set to be used
AC_ARG_WITH(simd-inst-set,
	AS_HELP_STRING([--with-simd-inst-set=set],[Select the set of SIMD instruction (avx [default], mmx or avx512)]),
	with_simd_inst_set="${withval}",
	with_simd_inst_set=avx)
case "$with_simd_inst_set" in
     mmx) CPPFLAGS_SIMD="-mmmx"
     	  SIMD_INST_SET=MMX;;
     avx) CPPFLAGS_SIMD="-mavx"
     	  SIMD_INST_SET=AVX;;
     avx512) CPPFLAGS_SIMD="-mavx512f"
     	     SIMD_INST_SET=AVX512;;
      *) AC_MSG_ERROR(["Unkwnown SIMD instruction set ${withval}"])
esac

#maybe we make a function for this
if test "$CXX" == "nvcc"
then
	CPPFLAGS="-Xcompiler $CPPFLAGS_SIMD -Xcompiler -mfma $CPPFLAGS"
else
	CPPFLAGS="$CPPFLAGS_SIMD -mfma $CPPFLAGS"
fi

SUMMARY_RESULT="$SUMMARY_RESULT
Simd Inst set      : $with_simd_inst_set"
AC_MSG_RESULT(with SIMD insttruction set ... ${with_simd_inst_set})
AC_DEFINE_UNQUOTED([SIMD_INST_SET],[$SIMD_INST_SET],[Instruction set to be used])

#add some flags which we will have to check again
if test "$CXX" == "nvcc"
then
	AC_DEFINE([USE_CUDA],1,"Using CUDA")
	CPPFLAGS="-dc --expt-relaxed-constexpr --expt-extended-lambda -x cu $CPPFLAGS"
fi

# Configure doxygen
DX_INIT_DOXYGEN($PACKAGE_NAME,[config/Doxyfile],[docs])
DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_MAN_FEATURE(ON)
DX_HTML_FEATURE(ON)
AC_SUBST([DOXYGEN_OUTPUT_DIR],[docs])

#result of conf to screen
AC_MSG_RESULT([
--------------------------------------------------
Configuration summary for $PACKAGE_NAME ($VERSION)
--------------------------------------------------
CC                  : $CC
CPPFLAGS            : $CPPFLAGS
CXXFLAGS            : $CXXFLAGS
CXX                 : $CXX
---
LDFLAGS             : $LDFLAGS
LIBS                : $LIBS
---
Installation prefix : $prefix
$LIBRARY_RESULT
$SUMMARY_RESULT
--------------------------------------------------
Configuration OK
--------------------------------------------------
])

#files to configure
AC_CONFIG_FILES(
	Makefile
	config/Doxyfile)

AC_OUTPUT
