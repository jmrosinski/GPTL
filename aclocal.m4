dnl Sets OMPCFLAGS, OMPDEFS.  Sets OMP to YES or NO
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
    AC_MSG_RESULT([$OMPCFLAGS])
  else
    AC_MSG_RESULT([not found])
    AC_MSG_WARN([Threaded tests may behave incorrectly])
    OMPCFLAGS=""
  fi
  CFLAGS="$OLDFLAGS"
])

dnl Sets OMPFFLAGS.  Sets FORTOMP to YES or NO
AC_DEFUN(UD_SET_OMP_F77,
[
  AC_LANG_PUSH(Fortran 77)
  OMPFFLAGS=""
  OLDFLAGS="$FFLAGS"
  FORTOMP="NO"

  AC_MSG_CHECKING([Fortran flags for openmp])

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
    AC_MSG_RESULT([$OMPFFLAGS])
  else
    AC_MSG_RESULT([not found])
    AC_MSG_WARN([Threaded tests may behave incorrectly])
    OMPFFLAGS=""
  fi
  FFLAGS="$OLDFLAGS"
  AC_LANG_POP(Fortran 77)
])

dnl Sets PTHREADCFLAGS, PTHREADDEFS.  Sets PTHREADS to YES or NO
AC_DEFUN(UD_SET_PTHREADS_C,
[
  OLDLDFLAGS="$LDFLAGS"
  PTHREADDEFS=""
  PTHREADCFLAGS="-lpthread"
  LDFLAGS="$OLDLDFLAGS $PTHREADCFLAGS"
  PTHREADS="NO"

  AC_MSG_CHECKING([C flags for pthreads])
  AC_TRY_LINK([#include <pthread.h>],[(void) pthread_self();],PTHREADS="YES",)

  if test "$PTHREADS" = "YES" ; then
    PTHREADDEFS=-DTHREADED_PTHREADS
    AC_MSG_RESULT([$PTHREADCFLAGS])
  else
    AC_MSG_RESULT([not found])
    AC_MSG_WARN([Threaded tests may behave incorrectly])
    PTHREADCFLAGS=""
  fi
  LDFLAGS="$OLDLDFLAGS"
])

dnl Modify FLDFLAGS only, not LDFLAGS in the end

AC_DEFUN(UD_SET_PTHREADS_F77,
[
  AC_LANG_PUSH(Fortran 77)
  OLDLDFLAGS="$LDFLAGS"
  PTHREADFFLAGS="-lpthread"
  LDFLAGS="$OLDLDFLAGS $PTHREADFFLAGS"
  PTHREADS="NO"

  AC_MSG_CHECKING([Fortran flags for pthreads])
  AC_TRY_LINK(,[pthread_self()],PTHREADS="YES";,)

  if test "$PTHREADS" = "YES" ; then
    THREADDEFS=-DTHREADED_PTHREADS
    AC_MSG_RESULT([$PTHREADFFLAGS])
  else
    AC_MSG_RESULT([not found])
    AC_MSG_WARN([Threaded Fortran tests may behave incorrectly])
    PTHREADFFLAGS=""
  fi
  AC_LANG_POP(Fortran 77)
  LDFLAGS="$OLDLDFLAGS"
])
