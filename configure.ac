#                                               -*- Autoconf -*-
## Process this file with autoconf to produce a configure script.

# This is part of Unidata's GPTL package. See the COPYRIGHT file for
# more information.
# Ed Hartnett, 5/18/18

# Specify minimum autoconf version.
AC_PREREQ([2.59])

# Initialize autoconf.
AC_INIT([GPTL], [2.0.0], [james.rosinski@noaa.gov])

# Keep libtool macros in an m4 directory.
AC_CONFIG_MACRO_DIR([m4])

# Find out about the host we're building on.
AC_CANONICAL_HOST

# Find out about the target we're building for.
AC_CANONICAL_TARGET

# Initialize automake.
AM_INIT_AUTOMAKE([foreign subdir-objects])

# Set up libtool.
LT_PREREQ([2.4])
LT_INIT()

# The config.h file will be created when configure script is run.
AC_CONFIG_HEADERS([config.h])

AC_LANG(C)
#AC_PROG_MAKE_SET
AC_PROG_CC()
#AC_PROG_FC()

dnl dnl This barfs when CC is pgcc and F77 pgf90
dnl AC_FC_FUNC(z_zz,z_zz)
dnl case $z_zz in
dnl     z_zz__)     echo "Fortran name mangling: UNDERSCORING=-DFORTRANDOUBLEUNDERSCORE" ;;
dnl     z_zz_)      echo "Fortran name mangling: UNDERSCORING=-DFORTRANUNDERSCORE" ;;
dnl     Z_ZZ)       echo "Fortran name mangling: UNDERSCORING=-DFORTRANCAPS" ;;
dnl     *)          echo "Fortran name mangling: UNDERSCORING=" ;;
dnl esac

dnl dnl For proper handling of const and inline
dnl AC_C_INLINE
dnl echo "Inlining: -Dinline=$ac_cv_c_inline"
dnl dnl AC_FUNC_VPRINTF
dnl AC_CHECK_LIB(papi,PAPI_library_init,[echo "PAPI library found: OK to set HAVE_PAPI=yes"],
dnl                                     [echo "PAPI library not found: HAVE_PAPI=no"])

dnl AC_CHECK_FUNC(backtrace_symbols,[echo "backtrace_symbols found: OK to set HAVE_BACKTRACE=yes"],
dnl 				[echo "backtrace_symbols NOT found: HAVE_BACKTRACE=no"])

dnl unset usempich;
dnl unset usempi;
dnl AC_CHECK_LIB(mpich,MPI_Init,[echo "libmpich.a found: OK to set HAVE_MPI=yes";usempich=yes],
dnl 	     AC_CHECK_LIB(mpi,MPI_Init,[echo "libmpi.a found: OK to set HAVE_MPI=yes OK";usempi=yes],[echo "MPI library not found: HAVE_MPI=no"]))
dnl AC_CHECK_FILE(/proc,[echo "/proc found: HAVE_SLASHPROC=yes"],
dnl 		    [echo "/proc not found: HAVE_SLASHPROC=no"])
dnl AC_CHECK_LIB(pthread,pthread_mutex_init,[echo "pthreads library found: OK to set PTHREADS=yes"],
dnl 					[echo "pthreads library not found: PTHREADS=no"])
AC_DEFINE([HAVE_SLASHPROC], [1], [some comment])

dnl if test -n "${usempich}" ; then
dnl AC_CHECK_LIB(mpich,iargc,[echo "iargc found in libmpich.a: OK to set HAVE_IARGCGETARG=yes"],	                   [echo "iargc not found in libmpich.a: HAVE_IARGCGETARG=no"])
dnl fi

dnl if test -n "${usempi}" ; then
dnl AC_CHECK_LIB(mpi,iargc,[echo "iargc found in libmpi.a: OK to set HAVE_IARGCGETARG=yes"],
dnl 		       [echo "iargc not found in libmpi.a: HAVE_IARGCGETARG=no"])
dnl fi

dnl AC_CHECK_SIZEOF([void *])
dnl if test "$ac_cv_sizeof_void_p" = 8; then
dnl   echo "Pointer size = 8 so BIT64=yes";
dnl elif test "$ac_cv_sizeof_void_p" = 4; then
dnl   echo "Pointer size = 4 so BIT64=no";
dnl fi

AC_CONFIG_FILES([Makefile
                 include/Makefile
                 test/Makefile
                 src/Makefile
                 src/clib/Makefile
		 ])
dnl AC_CONFIG_FILES([Makefile
dnl                  include/Makefile
dnl                  test/Makefile
dnl                  src/Makefile
dnl                  src/clib/Makefile
dnl                  src/flib/Makefile
dnl 		 ])

AC_OUTPUT()
