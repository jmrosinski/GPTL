dnl Process this file with autoconf to produce a configure script.
AC_INIT(private.h)
AC_LANG(C)

AC_PROG_MAKE_SET

#Get the OS type
OS="`uname | tr '[A-Z]' '[a-z]'`"
case "$OS" in
  linux)  OS=linux;;
  sunos)  OS=solaris ;;
  irix64) OS=irix ;;
  aix)    OS=aix;;
esac

dnl C and Fortran compilers
dnl Embedding AC_PROG_XX inside the shell code failed on SGI
if test "x$OS" = "xlinux"; then
  CCLIST="cc gcc g++ icc pgcc"
  FCLIST="g77 f77 pgf90 ifort lf95 g95"
elif test "x$OS" = "xirix"; then
  CCLIST="cc gcc g++"
  FCLIST="f90 g77 f77"
elif test "x$OS" = "xaix"; then
  CCLIST="xlc_r cc_r xlc cc gcc g++"
  FCLIST="xlf90_r xlf90"
elif test "x$OS" = "xsolaris"; then
  CCLIST="cc gcc g++"
  FCLIST="f90 g77 f77"
else
  CCLIST=""
  FCLIST=""
fi

AC_PROG_CC($CCLIST)
AC_PROG_F77($FCLIST)

AC_ARG_ENABLE(bit64,
            [--enable-bit64      Specify to build 64 bit library],
	    bit64=YES,bit64=NO)

BITEXT=

if test "$bit64" = YES; then
  BITEXT=64;
  if test "x$OS" = xirix ; then
     BITFLAGS="-64";
     CFLAGS="$CFLAGS $BITFLAGS";
     FFLAGS="$FFLAGS $BITFLAGS";
  elif test "x$OS" = xaix ; then
     BITFLAGS="-q64";
     CFLAGS="$CFLAGS $BITFLAGS";
     FFLAGS="$FFLAGS $BITFLAGS";
     ARFLAGS="-X 64";
  fi
fi

ARFLAGS="$ARFLAGS ruv";

dnl Whether to include PAPI lib
AC_ARG_ENABLE([papi],[--enable-papi       Specify to include PAPI library],
              [PAPI=YES;PAPIOBJS=gpt_papi.o],[PAPI=NO])

FORTDEFS=""
if test "$PAPI" = YES ; then
  AC_DEFINE([HAVE_PAPI])
  if test "x$PAPI_DIR" != x ; then
    PAPIPREP="-I$PAPI_DIR/include"
    LDFLAGS="$LDFLAGS -L$PAPI_DIR/lib"
  fi
dnl AIX needs pmapi
dnl AIX also passes defines through Fortran in a special way
  if test "x$OS" = xaix ; then
    FORTDEFS="-WF,-DHAVE_PAPI"
    LDFLAGS="$LDFLAGS -lpmapi"
  else
    FORTDEFS="-DHAVE_PAPI"
  fi
dnl will add -lpapi if successful
  AC_CHECK_LIB(papi$BITEXT, PAPI_library_init,,
	       [AC_MSG_ERROR([[not found. Rebuild w/ PAPI_DIR set or w/o --enable-papi]])])
fi

if test "x$OS" = xaix ; then
  FORTDEFS="$FORTDEFS -WF,-DAIX"
fi

dnl AC_ARG_ENABLE(papi,[--enable-papi],PAPIPREP="-I/home/rosinski/include -DHAVE_PAPI";PAPILIB="-L/home/rosinski/lib -lpapi",)

dnl Whether to enable debugging
AC_ARG_ENABLE(debug,[--enable-debug      Specify to enable debugging],
              [CFLAGS="$CFLAGS -g";
	       FFLAGS="$FFLAGS -g";
	       LDFLAGS="$LDFLAGS -g";
	       FLDFLAGS="$FLDFLAGS -g"],)

dnl Whether to enable optimization
AC_ARG_ENABLE(opt,[--enable-opt        Specify to optimize for speed],
              [CFLAGS="$CFLAGS -O";FFLAGS="$FFLAGS -O"],)

dnl need to add -qfixed to FFLAGS for Fortran tests on AIX
if test "$F77" = xlf90 ; then
  FFLAGS="$FFLAGS -qfixed"
fi

if test "$F77" = xlf90_r ; then
  FFLAGS="$FFLAGS -qfixed"
fi

dnl Whether and how to enable threading
dnl OMPDEFS, OMPCFLAGS
UD_SET_OMP_C
dnl PTHREADDEFS, PTHREADCFLAGS
UD_SET_PTHREADS_C
THREADDEFS=""
THREADFLAGS=""
AC_ARG_ENABLE(pthreads,[--enable-pthreads   Specify to thread with pthreads],
              [THREADDEFS="$PTHREADDEFS";THREADFLAGS="$PTHREADCFLAGS"],)

dnl openmp will override pthreads if both are specified
AC_ARG_ENABLE(openmp,[--enable-openmp     Specify to thread with OpenMP],
              [THREADDEFS="$OMPDEFS";THREADFLAGS="$OMPCFLAGS"],)
	                                
dnl OMPFFLAGS
UD_SET_OMP_F77

dnl How to mangle names:
AC_DEFINE(FDEFS)
AC_F77_FUNC(zzz,zzz)
case $zzz in
    zzz__)     FDEFS=-DFORTRANDOUBLEUNDERSCORE ;;
    zzz_)      FDEFS=-DFORTRANUNDERSCORE ;;
    ZZZ)       FDEFS=-DFORTRANCAPS ;;
esac

dnl Determine required Fortran load flags
dnl Dont do this now because of the above name mangling exersize.
dnl AC_F77_LIBRARY_LDFLAGS

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
AC_SUBST(THREADDEFS)
AC_SUBST(THREADFLAGS)
AC_SUBST(PAPIPREP)
AC_SUBST(PAPIOBJS)
AC_SUBST(ARFLAGS)
AC_SUBST(BITFLAGS)
AC_SUBST(FORTDEFS)
dnl for tests directory
AC_SUBST(OMPDEFS)
AC_SUBST(OMPCFLAGS)
AC_SUBST(OMPFFLAGS)
AC_SUBST(PTHREADDEFS)
AC_SUBST(PTHTEADCFLAGS)

AC_OUTPUT(Makefile tests/Makefile)
