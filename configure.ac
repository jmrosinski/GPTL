#                                               -*- Autoconf -*-
## Process this file with autoconf to produce a configure script.

# This is part of Unidata's GPTL package. See the COPYING file for
# more information.
# Ed Hartnett, 5/18/18

# Specify minimum autoconf version.
AC_PREREQ([2.59])

# Initialize autoconf.
AC_INIT([GPTL], [5.6.0], [james.rosinski@noaa.gov])

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

# Find the C compiler.
AC_PROG_CC()

# Find the Fortran compiler.
#AC_PROG_FC()

dnl dnl This barfs when CC is pgcc and F77 pgf90
dnl AC_FC_FUNC(z_zz,z_zz)
dnl case $z_zz in
dnl     z_zz__)     echo "Fortran name mangling: UNDERSCORING=-DFORTRANDOUBLEUNDERSCORE" ;;
dnl     z_zz_)      echo "Fortran name mangling: UNDERSCORING=-DFORTRANUNDERSCORE" ;;
dnl     Z_ZZ)       echo "Fortran name mangling: UNDERSCORING=-DFORTRANCAPS" ;;
dnl     *)          echo "Fortran name mangling: UNDERSCORING=" ;;
dnl esac

dnl For proper handling of const and inline
AC_C_CONST
AC_C_INLINE

# Does the user want to turn on nested OMP?
AC_MSG_CHECKING([whether nested OMP is to be enabled])
AC_ARG_ENABLE([nestedomp], [AS_HELP_STRING([--enable-nestedomp],
              [build with nested OMP capability])])
test "x$enable_nestedomp" = xyes || enable_nestedomp=no
AM_CONDITIONAL(ENABLE_NESTEDOMP, [test x$enable_nestedomp = xyes])
AC_MSG_RESULT($enable_nestedomp)

# Does the user want to turn on PMPI?
AC_MSG_CHECKING([whether PMPI is to be enabled])
AC_ARG_ENABLE([pmpi], [AS_HELP_STRING([--enable-pmpi],
              [build with PMPI capability])])
test "x$enable_pmpi" = xyes || enable_pmpi=no
AM_CONDITIONAL(ENABLE_PMPI, [test x$enable_pmpi = xyes])
AC_MSG_RESULT($enable_pmpi)

# Does the user want to test autoprofiling?
AC_MSG_CHECKING([whether autoprofile testing is to be enabled])
AC_ARG_ENABLE([autoprofile-testing], [AS_HELP_STRING([--enable-autoprofile-testing],
              [test autoprofileing])])
test "x$enable_autoprofile_testing" = xyes || enable_autoprofile_testing=no
AM_CONDITIONAL(TEST_AUTOPROFILE, [test x$enable_autoprofile_testing = xyes])
AC_MSG_RESULT($enable_autoprofile_testing)

# Check for papi library.
AC_CHECK_LIB([papi], [PAPI_library_init])
AM_CONDITIONAL([HAVE_PAPI], [test "x$ac_cv_lib_papi_PAPI_library_init" = xyes])
if test "x$ac_cv_lib_papi_PAPI_library_init" = xyes; then
   AC_DEFINE([HAVE_PAPI], [1], [PAPI library is present])
fi

# Check for existance of procfs, the proc file system.
AC_CHECK_FILE([/proc],
        [AC_DEFINE([HAVE_SLASHPROC], [1], [some comment])])

# Check for pthread lirbary.
AC_CHECK_LIB([pthread], [pthread_mutex_init])
if test "x$ac_cv_lib_pthread_pthread_mutex_init" = xyes; then
   AC_DEFINE([PTHREADS], [1], [pthreads library is present])
fi

# We need the math library for some tests.
AC_CHECK_LIB([m], [floor], [],
                  [AC_MSG_ERROR([Can't find or link to the math library.])])

# Check for function backtrace_symbols.
AC_CHECK_FUNC([backtrace_symbols],
        [AC_DEFINE([HAVE_BACKTRACE], [1], [backtrace_symbols function is present])])

# Do we have MPI?
AC_CHECK_FUNC([MPI_Init],
        [AC_DEFINE([HAVE_MPI], [1], [MPI is present])])
AM_CONDITIONAL([HAVE_MPI], [test "x$ac_cv_func_MPI_Init" = xyes])

# Check for function iargc, which may be part of MPI.
AC_CHECK_FUNC([iargc],
        [AC_DEFINE([HAVE_IARGCGETARG], [1], [backtrace_symbols function is present])])

# Check the size of a void pointer.
AC_CHECK_SIZEOF([void *])
if test "x$ac_cv_sizeof_void_p" = x8; then
   AC_DEFINE([BIT64], [1], [void pointer is 8 bytes])
fi

# Is doxygen installed?
AC_CHECK_PROGS([DOXYGEN], [doxygen])
if test -z "$DOXYGEN"; then
   AC_MSG_WARN([Doxygen not found - documentation will not be built])
fi

# This is a list of files to be built.
AC_CONFIG_FILES([Makefile
                 include/Makefile
                 ctests/Makefile
                 src/Makefile
                 src/clib/Makefile
                 bin/Makefile
		 ])

# Build the files listed above.
AC_OUTPUT()
