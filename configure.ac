# This is part of the GPTL package. See the COPYING file for
# more information.
# Ed Hartnett, 5/18/18

# Specify minimum autoconf version.
AC_PREREQ([2.59])

# Initialize autoconf.
AC_INIT([GPTL], [8.0.0], [rosinski@ucar.edu])

# Find out about the host we're building on.
AC_CANONICAL_HOST

# Find out about the target we're building for.
AC_CANONICAL_TARGET

# Initialize automake.
AM_INIT_AUTOMAKE([foreign subdir-objects])

# Set the compiler to C++
AC_LANG(C++)

# Find the C++ compiler.
AC_PROG_CXX()

# Keep libtool macros in an m4 directory.
AC_CONFIG_MACRO_DIR([m4])

# Set up libtool.
LT_PREREQ([2.4])
LT_INIT()

# The config.h file will be created when configure script is run.
AC_CONFIG_HEADERS([config.h])

# Only define HAVE_NANOTIME if this is a x86. It provides by far the finest grained,
# lowest overhead wallclock timer on that architecture.
# Set HAVE_NANOTIME on x86 systems only.
AC_MSG_CHECKING([whether x86 nanotime is available])
AS_CASE([$host], [*86*], [have_nanotime=yes], [have_nanotime=no])
if test "x$have_nanotime" = xyes; then
   AC_DEFINE([HAVE_NANOTIME], [1], [x86 nanotime capability is present])
  FCFLAGS="-DHAVE_NANOTIME $FCFLAGS"
fi
AC_MSG_RESULT($have_nanotime)

# Check the size of a void pointer to determine how to enable NANOTIME
AC_CHECK_SIZEOF([void *])
if test "x$ac_cv_sizeof_void_p" = x8; then
   AC_DEFINE([BIT64], [1], [void pointer is 8 bytes])
fi

AC_ARG_ENABLE([openmp], [AS_HELP_STRING([--disable-openmp],
              [Disable OpenMP support. May be necessary on e.g. MacOS using clang+gfortran])],,[enable_openmp=yes])

useomp=no;
if test "x$enable_openmp" != xno; then
# Enable OpenMP support if not told to skip, and compiler supports it
  AC_OPENMP()
  if test "x$ac_cv_prog_cxx_openmp" = xunsupported; then
    AC_MSG_NOTICE([OpenMP support not found. If CXX is Apple clang you could try CXXFLAGS=\"-Xpreprocessor -fopenmp\"])
  else
    useomp=yes;
    AC_DEFINE([THREADED_OMP], [1], [openmp support is present])
    CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"
  fi
fi
AM_CONDITIONAL([HAVE_OPENMP], [test "x$useomp" = xyes])

# May need this?
# CXXFLAGS="$CXXFLAGS -fpermissive"

# Does the user want to turn on nested OMP? Enable only if you will be
# profiling code regions inside of doubly-nested OpenMP regions. Using
# --enable-nestedomp can increase the cost of getting the thread
# number by about 50% even in regions which are not doubly-nested.
AC_MSG_CHECKING([whether nested OMP is to be enabled])
AC_ARG_ENABLE([nestedomp], [AS_HELP_STRING([--enable-nestedomp],
              [build with nested OMP capability])])

test "x$enable_nestedomp" = xyes || enable_nestedomp=no
# If nestedomp requested, verify basic OpenMP capability was there
if test "x$enable_nestedomp" = xyes; then
  if test "x$enable_openmp" != xyes; then
    AC_MSG_ERROR([Nested OMP capability requested but OpenMP disabled or not available])
  fi
  AC_DEFINE([ENABLE_NESTEDOMP], [1], [nested omp capability enabled])
fi
AC_MSG_RESULT($enable_nestedomp)
AM_CONDITIONAL(ENABLE_NESTEDOMP, [test x$enable_nestedomp = xyes])

# pthread library. default disabled
AC_ARG_ENABLE([pthread],
    AS_HELP_STRING([--enable-pthread], [Enable pthread capability instead of OpenMP]))

if test "x$enable_pthread" = xyes; then
  dnl Do the stuff needed for enabling the feature
  AC_MSG_CHECKING([for pthread support])
  AC_CHECK_LIB([pthread], [pthread_mutex_init])
  if test "x$ac_cv_lib_pthread_pthread_mutex_init" = xyes; then
     AC_DEFINE([THREADED_PTHREADS], [1], [pthreads library is present])
  fi
fi
AM_CONDITIONAL(HAVE_PTHREADS, [test "x$ac_cv_lib_pthread_pthread_mutex_init" = xyes])

# GPTL only works with EITHER OpenMP OR pthread lib. Barf on attempts to enable both
if test "x$enable_omp" = xyes; then
  if test "x$ac_cv_lib_pthread_pthread_mutex_init" = xyes; then
    AC_MSG_ERROR([OpenMP and pthreads cannot both be enabled: For pthreads use --disable-openmp --enable-pthread])
  fi
fi

# Does the user want to use double underscores for fortran wrappers?
AC_MSG_CHECKING([whether double underscore for Fortran wrappers should be enabled])
AC_ARG_ENABLE([double-underscore], [AS_HELP_STRING([--enable-double-underscore],
              [use double underscore for Fortran wrappers])])
test "x$enable_double_underscore" = xyes || enable_double_underscore=no
AC_MSG_RESULT($enable_double_underscore)
if test $enable_double_underscore = yes; then
   AC_DEFINE([FORTRANDOUBLEUNDERSCORE], [1], [use double underscore for Fortran wrappers])
else
   AC_DEFINE([FORTRANUNDERSCORE], [1], [use single underscore for Fortran wrappers])
fi

# --enable-debug means build with low optimization and add some code inside some #ifdef DEBUG
AC_ARG_ENABLE([debug], [AS_HELP_STRING([--enable-debug],
              [enable DEBUG ifdef])])
test "x$enable_debug" = xyes || enable_debug=no
if test $enable_debug = yes; then
   AC_DEFINE([DEBUG], [1], [set debug ifdef])
   CXXFLAGS="$CXXFLAGS -g -O0"
fi

AC_MSG_CHECKING([whether PAPI is to be enabled])
AC_ARG_ENABLE([papi], [AS_HELP_STRING([--enable-papi], [enable built-in support for papi library. Requires that PAPI is already installed])])
test "x$enable_papi" = xyes || enable_papi=no
AC_MSG_RESULT($enable_papi)

if test x$enable_papi = xyes; then
# Check for papi library.
  AC_CHECK_LIB([papi], [PAPI_library_init])
  AC_MSG_CHECKING([whether system can support PAPI])
  if test "x$ac_cv_lib_papi_PAPI_library_init" = xyes; then
    have_papi=yes
    # If we have PAPI library, check /proc/sys/kernel/perf_event_paranoid
    # to see if we have permissions.
    if test -f /proc/sys/kernel/perf_event_paranoid; then
      if test `cat /proc/sys/kernel/perf_event_paranoid` != 1; then
         AC_MSG_ERROR([PAPI library found, but /proc/sys/kernel/perf_event_paranoid != 1
                            try sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'])
      fi
    fi
    AC_DEFINE([HAVE_PAPI], [1], [PAPI library is present and usable])
    PAPI="papi"
    PAPI_LINK="-l${PAPI}"
  else
    have_papi=no
    PAPI=""
    PAPI_LINK=""
  fi
  AC_MSG_RESULT($have_papi)
  AC_SUBST([PAPI], [$PAPI])
fi
AM_CONDITIONAL([HAVE_PAPI], [test "x$have_papi" = xyes])

# Check for rt library. clock_gettime() in librt.a is an option for
# gathering wallclock time stats on some machines. Setting
# HAVE_LIBRT=yes enables this, but will probably require linking
# applications with -lrt
AC_CHECK_LIB([rt], [clock_gettime], [have_librt=yes], [have_librt=no],)
if test "$have_librt" = yes; then
  AC_DEFINE([HAVE_LIBRT],[1],[librt found])
  FCFLAGS="-DHAVE_LIBRT $FCFLAGS"
fi

# If getrusage is found then HAVE_GETRUSAGE will automatically be defined for obtaining RSS
AC_CHECK_FUNCS(getrusage)

# Check for existence of /proc, used for obtaining process size
# Must be disabled when cross-compiling
AC_ARG_ENABLE([slashproc], [AS_HELP_STRING([--disable-slashproc],
    	      [check for existence of /proc file system: must be disabled when cross-compiling])])

test "x$enable_slashproc" != xno && enable_slashproc=yes
if test "$enable_slashproc" = yes; then
  AC_CHECK_FILE([/proc],
        [AC_DEFINE([HAVE_SLASHPROC], [1], [/proc exists. Memory checking via /proc enabled])])
fi

# We need the math library for some tests.
AC_CHECK_LIB([m], [floor], [], [AC_MSG_ERROR([Can't find or link to the math library.])])

# For auto-profiling, libunwind is the default method to recover current function name.
# If disabled or not available, try backtrace. If neither are available, auto-profiling will 
# not be possible. To use auto auto-profiling with GNU or Intel compilers:
# Compile application code with -g and appropriate auto-instrumentation flag
# Probably the appropriate dynamic linking flag (maybe -rdynamic or -Bdynamic) also is needed
#
AC_ARG_ENABLE([libunwind],
    AS_HELP_STRING([--disable-libunwind], [Skip check for libunwind--check for backtrace instead],))

# Default is to enable auto-profiling capability via libunwind.
# Cannot have both unwind and backtrace methods enabled at once
found_libunwind=no
if test "x$enable_libunwind" != xno; then
  AC_MSG_CHECKING([for libunwind])
  AC_CHECK_LIB(unwind, [unw_backtrace], [found_libunwind=yes], [found_libunwind=no])
  AC_MSG_RESULT([$ac_cv_lib_unwind])
  if test "x$found_libunwind" = xyes; then
    AC_DEFINE([HAVE_LIBUNWIND], [1], [libunwind will be used])
    AC_MSG_NOTICE([libunwind enabled for auto-profiling: -lunwind and dynamic linking flag (see below) possibly also needed])
  fi
fi

found_backtrace=no
if test "x$found_libunwind" != xyes; then
  AC_MSG_NOTICE([libunwind disabled or not found])
  AC_MSG_CHECKING([for backtrace])
  AC_CHECK_FUNC([backtrace_symbols], found_backtrace=yes;)
  AC_MSG_RESULT([$ac_cv_func_backtrace_symbols])
  if test "x$found_backtrace" = xyes; then
    AC_DEFINE([HAVE_BACKTRACE], [1], [backtrace will be used])
    AC_MSG_NOTICE([backtrace enabled for auto-profiling: dynamic linking flag (see below) possibly also needed])
  fi
fi

# Above logic ensures that both have_libunwind and have_backtrace cannot both = yes
AM_CONDITIONAL([HAVE_LIBUNWIND], [test "x$found_libunwind" = xyes])
AM_CONDITIONAL([HAVE_BACKTRACE], [test "x$found_backtrace" = xyes])

# See if auto-instrumentation flag is available. If so, set INSTRFLAG for testing
if test "x$found_libunwind" = xyes || test "x$found_backtrace" = xyes; then
  AX_CHECK_COMPILE_FLAG([-finstrument-functions], [finstrf=yes], [finstrf=no])
  AX_CHECK_COMPILE_FLAG([-Minstrument:functions], [minstrf=yes], [minstrf=no])
fi

# Auto-instrumentation often requires a special link flag
rdynamic=yes
bdynamic=yes
AX_CHECK_LINK_FLAG(-rdynamic, [AC_SUBST([INSTR_LINK], [-rdynamic])], [rdynamic=no])
if test "$rdynamic" = yes; then
  INSTR_LINK="-rdynamic"
  AC_MSG_NOTICE([-rdynamic link flag may be needed for auto-instrumentation])
else
  AX_CHECK_LINK_FLAG(-Bdynamic, [AC_SUBST([INSTR_LINK], [-Bdynamic])], [bdynamic=no])
  if test "$bdynamic" = yes; then
    INSTR_LINK="-Bdynamic"
    AC_MSG_NOTICE([-Bdynamic link flag may be needed for auto-instrumentation])
  else
    AC_MSG_WARN([Dynamic linking flag not found: Auto-instrumentation may not work properly])
  fi
fi

AM_CONDITIONAL([HAVE_INSTRFLAG], [test "x$finstrf" = xyes || test "x$minstrf" = xyes])

if test "x$finstrf" = xyes; then
  INSTRFLAG="-finstrument-functions"
  AC_SUBST([INSTRFLAG],[$INSTRFLAG])
  AC_MSG_NOTICE([To auto-profile user code, Add compilation flag $INSTRFLAG to desired files])
elif test "x$minstrf" = xyes; then
  INSTRFLAG="-Minstrument:functions"
  AC_SUBST([INSTRFLAG],[$INSTRFLAG])
  AC_MSG_NOTICE([To auto-profile user code, Add compilation flag $INSTRFLAG to desired files])
fi

# Check for times.
AC_CHECK_FUNC([times],
        [AC_DEFINE([HAVE_TIMES], [1], [times function is available])])

# Check for gettimeofday.
AC_CHECK_FUNC([gettimeofday],
        [AC_DEFINE([HAVE_GETTIMEOFDAY], [1], [gettimeofday function is available])])

# Do we have MPI?
AC_CHECK_FUNC([MPI_Init], [have_libmpi=yes], [have_libmpi=no])
if test "x$have_libmpi" = xyes; then
# Modify FCFLAGS in case Fortran support is enabled
  FCFLAGS="-DHAVE_LIBMPI $FCFLAGS"
  AC_DEFINE([HAVE_LIBMPI], [1], [found MPI library])
# Need MPI_STATUS_SIZE for Fortran wrappers
  AC_CHECK_SIZEOF([int])
# Double brackets are important
  AC_CHECK_SIZEOF([MPI_Status],[],[[#include <mpi.h>]])
  if test "$ac_cv_sizeof_MPI_Status" != 0; then
    mpi_status_size_in_ints=$((ac_cv_sizeof_MPI_Status / ac_cv_sizeof_int));
# Use unquoted so can use a variable for value (2nd arg)
    AC_DEFINE_UNQUOTED([MPI_STATUS_SIZE_IN_INTS], [$mpi_status_size_in_ints], [size of status in MPI])
    AC_MSG_NOTICE([MPI_STATUS_SIZE_IN_INTS = $ac_cv_sizeof_MPI_Status / $ac_cv_sizeof_int = $mpi_status_size_in_ints])
  else
# TODO: The value is not always 5!!!
    AC_DEFINE([MPI_STATUS_SIZE_IN_INTS], [5], [size of status in MPI])
    AC_MSG_WARN([Could not determine MPI_STATUS_SIZE_IN_INTS -- GUESSING 5])
  fi
fi

AM_CONDITIONAL([HAVE_LIBMPI], [test "x$have_libmpi" = xyes])

# Did the user specify an MPI launcher other than mpiexec?
AC_MSG_CHECKING([whether a user specified program to run mpi programs])
AC_ARG_WITH([mpiexec],
              [AS_HELP_STRING([--with-mpiexec=<command>],
                              [Specify command to launch MPI parallel tests.])],
            [MPIEXEC=$with_mpiexec], [MPIEXEC=mpiexec])
AC_MSG_RESULT([$MPIEXEC])
if test "x$have_libmpi" = xyes; then
  AC_MSG_NOTICE([If \"make check\" encounters MPI failures, try configuring with --with-mpiexec=mpirun or srun,...])
fi
AC_SUBST([MPIEXEC], [$MPIEXEC])

# Do we have function MPI_Comm_f2c?
# For some reason this check trips up titan even though it exists. Just define it for now,
# since it has been part of the MPI standard for some time.
#AC_CHECK_FUNC([MPI_Comm_f2c], [have_mpi_comm_f2c=yes], [have_mpi_comm_f2c=no])
#if test "x$have_mpi_comm_f2c" = xyes; then
AC_DEFINE([HAVE_COMM_F2C], [1], [Hopefully MPI_Comm_f2c is present])
#fi

# Does the user want to turn on PMPI?
AC_MSG_CHECKING([whether PMPI is to be enabled])
AC_ARG_ENABLE([pmpi], [AS_HELP_STRING([--enable-pmpi], [build with PMPI support to auto-profile MPI calls])])
test "x$enable_pmpi" = xyes || enable_pmpi=no
AM_CONDITIONAL([ENABLE_PMPI], [test x$enable_pmpi = xyes])
if test "$enable_pmpi" = yes; then
   AC_DEFINE([ENABLE_PMPI], [1], [enable pmpi])
fi
AC_MSG_RESULT($enable_pmpi)

# See if Fortran support is requested (default yes). If so check for working compiler and add
# Fortran directories
AC_ARG_ENABLE([fortran], [AS_HELP_STRING([--disable-fortran],
              [Disable fortran support. Only do this if enabling it causes problems])],,[enable_fortran=yes])

# For gptl.pc
GPTL_LIBS="-lgptl"

if test "x$enable_fortran" != "xno"; then
  AC_LANG_PUSH(Fortran)
  AC_PROG_FC()
# For gptl.pc
  GPTL_LIBS="-lgptlf $GPTL_LIBS"
# Make sure this file is copied to build directories for tests to work.
  AC_CONFIG_LINKS([fortran/tests/gptlnl:fortran/tests/gptlnl])
# Test OMP for Fortran and set flags accordingly, only if OMP support was enabled above for C
# AC_OPENMP sets OPENMP_FCFLAGS not OPENMP_FFLAGS
  have_fort_omp=no
  if test "x$useomp" = "xyes"; then
    AC_OPENMP()
    if test "x$ac_cv_prog_fc_openmp" = xunsupported; then
      AC_MSG_NOTICE([OpenMP Fortran support not found. Fortran OMP tests will not be exercised but GPTL OMP support is there])
    else
      have_fort_omp=yes
      FCFLAGS="-DTHREADED_OMP $FCFLAGS $OPENMP_FCFLAGS"
    fi
  else
    AC_MSG_NOTICE([Skipping check for Fortran OMP support because it was not found or disabled for C])
  fi
  AM_CONDITIONAL([HAVE_FORT_OPENMP], [test "x$have_fort_omp" = xyes])

# Modify FCFLAGS for MPI
  if test "x$have_libmpi" = xyes; then
    FCFLAGS="-DHAVE_LIBMPI $FCFLAGS"
  fi
# Modify FCFLAGS for PAPI
  if test "x$have_papi" = xyes; then
    FCFLAGS="-DHAVE_PAPI $FCFLAGS"
  fi
  
# This is a list of files to be built when Fortran support enabled
  AC_CONFIG_FILES([fortran/Makefile
                   fortran/include/Makefile
                   fortran/src/Makefile
                   fortran/tests/Makefile
	  	 ])
  AC_CONFIG_FILES([fortran/tests/run_par_pmpi_test.sh], [chmod ugo+x fortran/tests/run_par_pmpi_test.sh])
  AC_CONFIG_FILES([fortran/tests/run_par_summary_test.sh], [chmod ugo+x fortran/tests/run_par_summary_test.sh])
  AC_LANG_POP()
fi
AM_CONDITIONAL([HAVE_FORTRAN], [test x$enable_fortran = "xyes"])

# For gptl.pc
if test "x$found_libunwind" = xyes; then
  GPTL_LIBS="$GPTL_LIBS -lunwind"
fi
LDFLAGS=$PAPI_LINK
LDFLAGS_PC="$GPTL_LIBS $PAPI_LINK $LDFLAGS $INSTR_LINK"
AC_SUBST([LDFLAGS], [$LDFLAGS])
AC_SUBST([LDFLAGS_PC], [$LDFLAGS_PC])
AC_SUBST([OPENMP_CXXFLAGS], [$OPENMP_CXXFLAGS])
AC_SUBST([OPENMP_FCFLAGS], [$OPENMP_FCFLAGS])

# Files to be built
AC_CONFIG_FILES([Makefile
                 gptl.pc
                 include/Makefile
                 tests/Makefile
                 src/Makefile
                 bin/Makefile
	       ])
AC_CONFIG_FILES([tests/run_par_summary_test.sh], [chmod ugo+x tests/run_par_summary_test.sh])
AC_CONFIG_FILES([tests/run_par_pmpi_test.sh], [chmod ugo+x tests/run_par_pmpi_test.sh])
AC_CONFIG_FILES([tests/run_par_global_test.sh], [chmod ugo+x tests/run_par_global_test.sh])

# Is doxygen installed?
AC_CHECK_PROGS([DOXYGEN], [doxygen])
if test -z "$DOXYGEN"; then
   AC_MSG_WARN([Doxygen not found - documentation will not be built])
fi

# Build the files listed above.
AC_OUTPUT()
