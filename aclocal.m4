AC_DEFUN(UD_SET_OMP_C,
[
  THREADFLAGS=""
  THREADDEFS=""
  OLDFLAGS="$CFLAGS"
  OMP="NO"

  AC_MSG_CHECKING([C flags for openmp])

  if test "$OMP" = "NO" ; then
    THREADFLAGS="-qsmp=omp"
    CFLAGS="$OLDFLAGS $THREADFLAGS"
    AC_TRY_LINK([#include <omp.h>],[(void) omp_get_max_threads();],OMP="YES",)
  fi

  if test "$OMP" = "NO" ; then
    THREADFLAGS="-mp"
    CFLAGS="$OLDFLAGS $THREADFLAGS"
    AC_TRY_LINK([#include <omp.h>],[(void) omp_get_max_threads();],OMP="YES",)
  fi

  if test "$OMP" = "NO" ; then
    THREADFLAGS="-openmp"
    CFLAGS="$OLDFLAGS $THREADFLAGS"
    AC_TRY_LINK([#include <omp.h>],[(void) omp_get_max_threads();],OMP="YES",)
  fi

  if test "$OMP" = "YES" ; then
    THREADDEFS=-DTHREADED_OMP
    AC_MSG_RESULT($THREADFLAGS works)
  else
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([quitting.  Rerun configure without --enable-openmp])
  fi
  CFLAGS="$OLDFLAGS"
])

AC_DEFUN(UD_SET_OMP_F77,
[
  AC_LANG_PUSH(Fortran 77)
  FTHREADFLAGS=""
  OLDFLAGS="$FFLAGS"
  FORTOMP="NO"

  AC_MSG_CHECKING([Fortran openmp threading])

  FTHREADFLAGS="-mp"
  FFLAGS="$OLDFLAGS $FTHREADFLAGS"
  AC_TRY_LINK(,[      call omp_get_max_threads()],FORTOMP="YES",)

  if test "$FORTOMP" = "NO" ; then
    FTHREADFLAGS="-qsmp=omp"
    FFLAGS="$OLDFLAGS $FTHREADFLAGS"
    AC_TRY_LINK(,[      call omp_get_max_threads()],FORTOMP="YES",)
  fi

  if test "$FORTOMP" = "NO" ; then
    FTHREADFLAGS="-omp"
    FFLAGS="$OLDFLAGS $FTHREADFLAGS"
    AC_TRY_LINK(,[      call omp_get_max_threads()],FORTOMP="YES",)
  fi

  if test "$FORTOMP" = "NO" ; then
    FTHREADFLAGS="-openmp"
    FFLAGS="$OLDFLAGS $FTHREADFLAGS"
    AC_TRY_LINK(,[      call omp_get_max_threads()],FORTOMP="YES",)
  fi

  if test "$FORTOMP" = "YES" ; then
    AC_MSG_RESULT($FTHREADFLAGS)
  else
    AC_MSG_RESULT([not found])
    AC_MSG_WARN([threaded Fortran tests may fail])
  fi
  FFLAGS="$OLDFLAGS"
  AC_LANG_POP(Fortran 77)
])

AC_DEFUN(UD_SET_PTHREADS_C,
[
  OLDLDFLAGS="$LDFLAGS"
  THREADFLAGS="-lpthread"
  LDFLAGS="$OLDLDFLAGS $THREADFLAGS"
  PTHREADS="NO"

  AC_MSG_CHECKING([pthreads under C])
  AC_TRY_LINK([#include <pthread.h>],[(void) pthread_self();],PTHREADS="YES",)

  if test "$PTHREADS" = "YES" ; then
    THREADDEFS=-DTHREADED_PTHREADS
    AC_MSG_RESULT($THREADFLAGS)
  else
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([quitting.  Rerun configure without --enable-pthreads])
  fi
])

dnl Modify FLDFLAGS only, not LDFLAGS in the end

AC_DEFUN(UD_SET_PTHREADS_F77,
[
  AC_LANG_PUSH(Fortran 77)
  OLDLDFLAGS="$LDFLAGS"
  FTHREADFLAGS="-lpthread"
  LDFLAGS="$OLDLDFLAGS $FTHREADFLAGS"
  PTHREADS="NO"

  AC_MSG_CHECKING([pthreads under Fortran])
  AC_TRY_LINK(,[pthread_self()],PTHREADS="YES";FLDFLAGS="$LDFLAGS",)

  if test "$PTHREADS" = "YES" ; then
    THREADDEFS=-DTHREADED_PTHREADS
    AC_MSG_RESULT($FTHREADFLAGS)
  else
    AC_MSG_RESULT([not found])
  fi
  AC_LANG_POP(Fortran 77)
  LDFLAGS="$OLDLDFLAGS"
])
