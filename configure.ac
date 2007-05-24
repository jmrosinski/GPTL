dnl Process this file with autoconf to produce a configure script.
AC_INIT(private.h)
AC_LANG(C)

AC_PROG_MAKE_SET

dnl Get the OS type
OS="`uname | tr '[a-z]' '[A-Z]'`"
CAMFLAGS="-D$OS"
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
  FCLIST="g77 f77 pgf90 ifort lf95 gfortran g95"
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

FFLAGS=
LDFLAGS=
INSTRFLAGS=
AC_PROG_CC($CCLIST)
dnl Whether to enable optimization
AC_ARG_ENABLE(opt,[--enable-opt        Optimize for speed],
              [CFLAGS="-O3";FFLAGS="-O"],)

dnl Whether to enable debugging
AC_ARG_ENABLE(debug,[--enable-debug      Enable debugging],
              [CFLAGS="-g";FFLAGS="-g";LDFLAGS="-g"],)

dnl For gcc only
if test "x$GCC" = xyes; then
   INSTRFLAGS="-finstrument-functions";
   CFLAGS="$CFLAGS -finline-functions -Winline"
fi

AC_PROG_F77($FCLIST)

dnl Whether to target 64-bit addressing
AC_ARG_ENABLE(bit64,
            [--enable-bit64      Build 64 bit library],
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

dnl Check for various underlying timing routines
AC_CHECK_LIB(rt, clock_gettime,,)
AC_CHECK_LIB(mpi, MPI_Wtime,,
             AC_CHECK_LIB(mpich, MPI_Wtime,,))
AC_CHECK_FUNCS(gettimeofday times,,)

dnl One of the Fortran tests has ifdef AIX and defines are funky on AIX Fortran
if test "x$OS" = xaix ; then
  FORTDEFS="$FORTDEFS -WF,-DAIX"
fi

dnl check for nanotime
AC_ARG_ENABLE(nanotime,[--enable-nanotime   Replace gettimeofday with native IA32 or IA64],
              AC_DEFINE([HAVE_NANOTIME]),)

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

dnl This barfs when CC is pgcc and F77 pgf90
AC_F77_FUNC(z_zz,z_zz)
case $z_zz in
    z_zz__)     FDEFS=-DFORTRANDOUBLEUNDERSCORE ;;
    z_zz_)      FDEFS=-DFORTRANUNDERSCORE ;;
    Z_ZZ)       FDEFS=-DFORTRANCAPS ;;
esac

dnl Determine required Fortran load flags
dnl Dont do this now because of the above name mangling exersize.
dnl AC_F77_LIBRARY_LDFLAGS

dnl For proper handling of const and inline
AC_C_CONST
AC_C_INLINE

dnl Checks for library functions.
AC_FUNC_VPRINTF

dnl Checks for AR
AC_CHECK_PROG(AR,ar)

AC_SUBST(FDEFS)
AC_SUBST(THREADDEFS)
AC_SUBST(THREADFLAGS)
AC_SUBST(FTHREADFLAGS)
AC_SUBST(PAPIPREP)
AC_SUBST(ARFLAGS)
AC_SUBST(BITFLAGS)
AC_SUBST(FORTDEFS)
AC_SUBST(INSTRFLAGS)
AC_SUBST(CAMFLAGS)

AC_OUTPUT(Makefile ctests/Makefile ftests/Makefile)

dnl Print warnings if appropriate.
if test "$OMP" = "NO" ; then
  AC_MSG_WARN([GPTL will be built WITHOUT the requested openmp threading. See config.log])
elif test "$PTHREADS" = "NO" ; then
  AC_MSG_WARN([GPTL will be built WITHOUT the requested pthreads threading. See config.log])
fi
