dnl Process this file with autoconf to produce a configure script.
AC_INIT(private.h)
AC_LANG(C)

AC_PROG_MAKE_SET

dnl Get the OS type
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
  AC_DEFINE([LINUX])
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

dnl Whether to target 64-bit addressing
AC_ARG_ENABLE(bit64,
            [--enable-bit64      Build 64 bit library],
	    bit64=YES,bit64=NO)

BITEXT=
CFLAGS=

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
  elif test "x$OS" = xlinux ; then
     CFLAGS="$CFLAGS -DBIT64";
  fi
fi

dnl Archive flags
ARFLAGS="$ARFLAGS ruv";

dnl Whether to include PAPI lib and how to do so
FORTDEFS=""
AC_ARG_WITH(papi,[--with-papi=<path>  Specify path to root of PAPI installation],
            [PAPI_ROOT="$withval";
             PAPI=YES;
	     PAPIOBJS=gptl_papi.o;
             PAPIPREP="-I$withval/include";
	     LDFLAGS="$LDFLAGS -L$PAPI_ROOT/lib"],

	     [PAPI=NO])

if test "$PAPI" = YES ; then
  AC_DEFINE([HAVE_PAPI])
dnl AIX needs pmapi
dnl AIX also passes defines through Fortran in a special way
  if test "x$OS" = xaix ; then
    FORTDEFS="-WF,-DHAVE_PAPI"
    LDFLAGS="$LDFLAGS -lpmapi"
  else
    FORTDEFS="-DHAVE_PAPI"
  fi
dnl will add -lpapi or -lpapi64 if successful
  AC_CHECK_LIB(papi$BITEXT, PAPI_library_init,,
               AC_CHECK_LIB(papi, PAPI_library_init,,
	       [AC_MSG_ERROR([[not found. Rebuild using --with-papi or w/o --enable-papi]])]))
fi

dnl One of the Fortran tests has ifdef AIX and defines are funky on AIX Fortran
if test "x$OS" = xaix ; then
  FORTDEFS="$FORTDEFS -WF,-DAIX"
fi

dnl Whether to enable debugging
AC_ARG_ENABLE(debug,[--enable-debug      Enable debugging],
              [CFLAGS="$CFLAGS -g"
	       FFLAGS="$FFLAGS -g"
	       LDFLAGS="$LDFLAGS -g"
	       FLDFLAGS="$FLDFLAGS -g"],)

dnl Whether to enable debugging
AC_ARG_ENABLE(nanotime,[--enable-nanotime   Replace gettimeofday with native IA32 or IA64],
              AC_DEFINE([NANOTIME]),)

dnl Add GNU-specific options

if test "$CC" = gcc ; then
  CFLAGS="$CFLAGS -finline-functions -Winline"
fi

if test "$CC" = g++ ; then
  CFLAGS="$CFLAGS -finline-functions -Winline"
fi

dnl Whether to enable optimization
AC_ARG_ENABLE(opt,[--enable-opt        Optimize for speed],
              [CFLAGS="$CFLAGS -O3";FFLAGS="$FFLAGS -O"],)

dnl Whether to use numeric rather than character timers
AC_ARG_ENABLE(numeric,[--enable-numeric    Numeric rather than character timers ALSO adds __cyg_profile_func auto-instrumentation capability],
              [CFLAGS="$CFLAGS -DNUMERIC_TIMERS"],)

dnl need to add -qfixed to FFLAGS for Fortran tests on AIX
if test "$F77" = xlf90 ; then
  FFLAGS="$FFLAGS -qfixed"
fi

if test "$F77" = xlf90_r ; then
  FFLAGS="$FFLAGS -qfixed"
fi

dnl Whether and how to enable threading
THREADDEFS=""
THREADFLAGS=""
FTHREADFLAGS=""
OMP=""
PTHREADS=""
dnl OMPDEFS, OMPCFLAGS, OMP
AC_ARG_ENABLE(openmp,[--enable-openmp     OpenMP threading (overrides pthreads)],
              UD_SET_OMP_C,)
if test "$OMP" = YES ; then
  THREADDEFS="$OMPDEFS"
  THREADFLAGS="$OMPCFLAGS"
  UD_SET_OMP_F77
  FTHREADFLAGS="$OMPFFLAGS"
else
dnl PTHREADDEFS, PTHREADCFLAGS, PTHREADS
  AC_ARG_ENABLE(pthreads,[--enable-pthreads   pthreads threading],UD_SET_PTHREADS_C,)
  if test "$PTHREADS" = YES ; then
    THREADDEFS="$PTHREADDEFS"
    THREADFLAGS="$PTHREADCFLAGS"
    UD_SET_PTHREADS_F77
    FTHREADFLAGS="$PTHREADFFLAGS"
  fi
fi

dnl How to mangle Fortran names:
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

dnl For proper handling of const and inline
AC_C_CONST
AC_C_INLINE

dnl Checks for library functions.
AC_FUNC_VPRINTF
AC_CHECK_FUNCS(gettimeofday times,,AC_MSG_ERROR([not found. giving up]))

AC_SUBST(FDEFS)
AC_SUBST(THREADDEFS)
AC_SUBST(THREADFLAGS)
AC_SUBST(FTHREADFLAGS)
AC_SUBST(PAPIPREP)
AC_SUBST(PAPIOBJS)
AC_SUBST(ARFLAGS)
AC_SUBST(BITFLAGS)
AC_SUBST(FORTDEFS)

AC_OUTPUT(Makefile tests/Makefile)

dnl Print warnings if appropriate.
if test "$OMP" = "NO" ; then
  AC_MSG_WARN([GPTL will be built WITHOUT the requested openmp threading. See config.log])
elif test "$PTHREADS" = "NO" ; then
  AC_MSG_WARN([GPTL will be built WITHOUT the requested pthreads threading. See config.log])
fi
