dnl Process this file with autoconf to produce a configure script.
AC_INIT(private.h)
AC_LANG(C)

dnl Checks for programs.
AC_PROG_CC(pgcc cc xlc_r cc_r xlc g++ gcc icc)

dnl Added by me:
dnl What fortran compiler to use:
AC_PROG_F77(pgf90 xlf90_r f90 lf95 g95 ifort g77 f77)

dnl Whether to enable debugging
AC_ARG_ENABLE(debug,[--enable-debug],CFLAGS=-g;FFLAGS=-g,)

dnl Whether to enable optimization
AC_ARG_ENABLE(opt,[--enable-opt],CFLAGS=-O;FFLAGS=-O,)

dnl Whether and how to enable threading
AC_ARG_ENABLE(openmp,[--enable-openmp],UD_SET_OMP_C,)
if test "$OMP" = YES ; then
  UD_SET_OMP_F77
fi

dnl How to mangle names:
AC_DEFINE(FDEFS)
AC_F77_FUNC(zzz,zzz)
case $zzz in
    zzz__)     FDEFS=-DFORTRANDOUBLEUNDERSCORE ;;
    zzz_)      FDEFS=-DFORTRANUNDERSCORE ;;
    ZZZ)       FDEFS=-DFORTRANCAPS ;;
esac

dnl Determine required Fortran load flags
AC_F77_LIBRARY_LDFLAGS

dnl Checks for libraries.
dnl Replace `main' with a function in -lm:
AC_CHECK_LIB(m, sqrt)

dnl Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS(sys/time.h unistd.h)

dnl Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_HEADER_TIME

dnl Checks for library functions.
AC_FUNC_VPRINTF
AC_CHECK_FUNCS(gettimeofday)

AC_SUBST(FDEFS)
AC_SUBST(FLDFLAGS)
AC_SUBST(CDEFS)
AC_SUBST(THREADDEFS)
AC_OUTPUT(Makefile tests/Makefile)
