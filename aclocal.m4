AC_DEFUN(UD_SET_OMP_C,
[
  OMPCFLAGS=""
  OMPDEFS=""
  OLDFLAGS="$CFLAGS"
  OMP="NO"

  AC_MSG_CHECKING([C flags for openmp])

  if test "$OMP" = "NO" ; then
    OMPCFLAGS="-qsmp=omp"
    CFLAGS="$OLDFLAGS $OMPCFLAGS"
    AC_TRY_LINK([#include <omp.h>],[(void) omp_get_max_threads();],OMP="YES",)
  fi

  if test "$OMP" = "NO" ; then
    OMPCFLAGS="-mp"
    CFLAGS="$OLDFLAGS $OMPCFLAGS"
    AC_TRY_LINK([#include <omp.h>],[(void) omp_get_max_threads();],OMP="YES",)
  fi

  if test "$OMP" = "NO" ; then
    OMPCFLAGS="-openmp"
    CFLAGS="$OLDFLAGS $OMPCFLAGS"
    AC_TRY_LINK([#include <omp.h>],[(void) omp_get_max_threads();],OMP="YES",)
  fi

  if test "$OMP" = "YES" ; then
    OMPDEFS=-DTHREADED_OMP
    AC_MSG_RESULT($OMPCFLAGS works)
  else
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([quitting.  Rerun configure without --enable-openmp])
  fi
  CFLAGS="$OLDFLAGS"
])

AC_DEFUN(UD_SET_OMP_F77,
[
  AC_LANG_PUSH(Fortran 77)
  OMPFFLAGS=""
  OLDFLAGS="$FFLAGS"
  FORTOMP="NO"

  AC_MSG_CHECKING([Fortran openmp threading])

  OMPFFLAGS="-mp"
  FFLAGS="$OLDFLAGS $OMPFFLAGS"
  AC_TRY_LINK(,[      call omp_get_max_threads()],FORTOMP="YES",)

  if test "$FORTOMP" = "NO" ; then
    OMPFFLAGS="-qsmp=omp"
    FFLAGS="$OLDFLAGS $OMPFFLAGS"
    AC_TRY_LINK(,[      call omp_get_max_threads()],FORTOMP="YES",)
  fi

  if test "$FORTOMP" = "NO" ; then
    OMPFFLAGS="-omp"
    FFLAGS="$OLDFLAGS $OMPFFLAGS"
    AC_TRY_LINK(,[      call omp_get_max_threads()],FORTOMP="YES",)
  fi

  if test "$FORTOMP" = "NO" ; then
    OMPFFLAGS="-openmp"
    FFLAGS="$OLDFLAGS $OMPFFLAGS"
    AC_TRY_LINK(,[      call omp_get_max_threads()],FORTOMP="YES",)
  fi

  if test "$FORTOMP" = "YES" ; then
    AC_MSG_RESULT($OMPFFLAGS)
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
  PTHREADCFLAGS="-lpthread"
  LDFLAGS="$OLDLDFLAGS $PTHREADCFLAGS"
  PTHREADS="NO"

  AC_MSG_CHECKING([pthreads under C])
  AC_TRY_LINK([#include <pthread.h>],[(void) pthread_self();],PTHREADS="YES",)

  if test "$PTHREADS" = "YES" ; then
    THREADDEFS=-DTHREADED_PTHREADS
    AC_MSG_RESULT($PTHREADCFLAGS)
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
  PTHREADFFLAGS="-lpthread"
  LDFLAGS="$OLDLDFLAGS $PTHREADFFLAGS"
  PTHREADS="NO"

  AC_MSG_CHECKING([pthreads under Fortran])
  AC_TRY_LINK(,[pthread_self()],PTHREADS="YES";FLDFLAGS="$LDFLAGS",)

  if test "$PTHREADS" = "YES" ; then
    THREADDEFS=-DTHREADED_PTHREADS
    AC_MSG_RESULT($PTHREADFFLAGS)
  else
    AC_MSG_RESULT([not found])
  fi
  AC_LANG_POP(Fortran 77)
  LDFLAGS="$OLDLDFLAGS"
  AC_SUBST(FLDFLAGS)
])
