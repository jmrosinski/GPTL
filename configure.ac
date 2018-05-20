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

# Check for papi library.
AC_CHECK_LIB([papi], [PAPI_library_init],
                     [AC_DEFINE([HAVE_PAPI], [1], [PAPI library is present])])

# Check for existance of procfs, the proc file system.
AC_CHECK_FILE([/proc],
        [AC_DEFINE([HAVE_SLASHPROC], [1], [some comment])])

# Check for pthread lirbary.
AC_CHECK_LIB([pthread], [pthread_mutex_init],
                        [AC_DEFINE([PTHREADS], [1], [pthreads library is present])])

# Check for function backtrace_symbols.
AC_CHECK_FUNC([backtrace_symbols],
        [AC_DEFINE([HAVE_BACKTRACE], [1], [backtrace_symbols function is present])])

# Do we have MPI?
AC_CHECK_FUNC([MPI_Init],
        [AC_DEFINE([HAVE_MPI], [1], [MPI is present])])

# Check for function iargc, which may be part of MPI.
AC_CHECK_FUNC([iargc],
        [AC_DEFINE([HAVE_IARGCGETARG], [1], [backtrace_symbols function is present])])

# Check the size of a void pointer.
AC_CHECK_SIZEOF([void *])
if test "x$ac_cv_sizeof_void_p" = x8; then
   AC_DEFINE([BIT64], [1], [void pointer is 8 bytes])
fi

# Is doxygen installed? If so, have configure construct the Doxyfile.
AC_CHECK_PROGS([DOXYGEN], [doxygen])
if test -z "$DOXYGEN"; then
   AC_MSG_WARN([Doxygen not found - documentation will not be built])
fi

# This is a list of files to be built.
AC_CONFIG_FILES([Makefile
                 include/Makefile
                 test/Makefile
                 src/Makefile
                 src/clib/Makefile
		 ])

# Build the files listed above.
AC_OUTPUT()
