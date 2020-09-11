# This is part of the GPTL package. See the COPYING file for
# more information.
# Ed Hartnett, 5/18/18

# Specify minimum autoconf version.
AC_PREREQ([2.59])

# Initialize autoconf.
AC_INIT([GPTL], [8.0.3], [jmrosinski@gmail.com])

# Find out about the host we're building on.
AC_CANONICAL_HOST

# Find out about the target we're building for.
AC_CANONICAL_TARGET

# Initialize automake.
AM_INIT_AUTOMAKE([foreign subdir-objects])

# Set the language as C
AC_LANG(C)

# Find the C compiler.
AC_PROG_CC()

# Keep libtool macros in an m4 directory.
AC_CONFIG_MACRO_DIR([m4])

# Set up libtool.
LT_PREREQ([2.4])
LT_INIT()

# The config.h file will be created when configure script is run.
AC_CONFIG_HEADERS([config.h])

# These ensure proper handling of const and inline.
AC_C_CONST
AC_C_INLINE

# Embed git version info, or release tag if not cloned
if test -d .git; then
  AC_CHECK_PROG(havegit, git, yes, no)
  if test "x$havegit" = xyes; then
    AC_MSG_CHECKING([for GPTL version info encoded into git])
    git_versioninfo=$(git describe --abbrev=4 --dirty --always --tags)
    AC_MSG_RESULT([$git_versioninfo])
  else
    git_versioninfo=NO_GIT
  fi
  AC_CHECK_PROG(git_versioninfo, git, $(git describe --abbrev=4 --dirty --always --tags), NO_GIT)
  AC_DEFINE_UNQUOTED([GPTL_VERSIONINFO], ["$git_versioninfo"], [use git describe for version info])
else
  AC_MSG_CHECKING([for package info for package version info since this is not a git repo])
  AC_DEFINE([GPTL_VERSIONINFO], ["AC_PACKAGE_VERSION"], [use configure for version info])
  AC_MSG_RESULT(AC_PACKAGE_VERSION)
fi

# Only define HAVE_NANOTIME if this is a x86. It provides by far the finest grained,
# lowest overhead wallclock timer on that architecture.
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

# Check for enabling gather/print stats on extra comparisons due to collisions
# Default disabled due to extra overhead
AC_ARG_ENABLE([collide],
  AS_HELP_STRING([--enable-collide], [Include stats on extra comparisons due to hash collisions]))
AS_IF([test "x$enable_collide" = xyes], [
  AC_DEFINE([COLLIDE], [1], [Include stats on extra comparisons due to hash collisions])
])

# Check whether to inline threading routines rather than compile separately
# Default enabled
AC_ARG_ENABLE([inline-threading],
  AS_HELP_STRING([--disable-inline-threading], [Do not inline threading routines]))
AS_IF([test "x$enable_inline_threading" != xno], [
  enable_inline_threading=yes
  AC_DEFINE([INLINE_THREADING], [1], [Use \#include to inline threading routines into gptl.c])
  AC_MSG_NOTICE([Enabling inline threading for efficiency. Use --disable-inline-threading if \"make\" fails])
  AC_MSG_NOTICE([Compiler warnings re: inline function GPTLget_threadnum declared but never defined may occur. They can be safely ignored])
])
AM_CONDITIONAL([INLINE_THREADING], [test "x$enable_inline_threading" = xyes])

# Check for enabling OpenMP support
# Default enabled
AC_ARG_ENABLE([openmp], [AS_HELP_STRING([--disable-openmp],
              [Disable OpenMP support. May be necessary on e.g. MacOS using clang+gfortran])])
useomp=no;
AS_IF([test "x$enable_openmp" != xno], [
# Enable OpenMP support if not told to skip, and compiler supports it
  AC_OPENMP()
  if test "x$ac_cv_prog_c_openmp" = xunsupported; then
    AC_MSG_NOTICE([OpenMP support not found. If CC is Apple clang you could try CFLAGS=\"-Xpreprocessor -fopenmp\"])
  else
    useomp=yes;
    AC_DEFINE([THREADED_OMP], [1], [openmp support is present])
    CFLAGS="$CFLAGS $OPENMP_CFLAGS"
  fi
])
AM_CONDITIONAL([HAVE_OPENMP], [test "x$useomp" = xyes])

# Whether to enable GPTL to check for nested OMP constructs and do the right thing.
# --enable-nestedomp can increase the cost of getting the thread number substantially,
# even in regions which are not nested. Currently with --enable-nestedomp, GPTL allows
# at most doubly-nested OpenMP regions.
# Default disabled
supportnesting=no
AC_MSG_CHECKING([whether nested OMP is to be enabled])
AC_ARG_ENABLE([nestedomp], [AS_HELP_STRING([--enable-nestedomp],
              [Build with nested OMP capability])])
AS_IF([test "x$enable_nestedomp" = "xyes"], [
  AC_DEFINE([ENABLE_NESTEDOMP], [1], [nested omp capability enabled])
  supportnesting=yes
])

# If nestedomp requested, verify basic OpenMP capability was there
if test "x$supportnesting" = xyes; then
  if test "x$useomp" != xyes; then
    AC_MSG_ERROR([Nested OMP capability requested but OpenMP disabled or not available])
  fi
  AC_DEFINE([ENABLE_NESTEDOMP], [1], [nested omp capability enabled])
fi
AC_MSG_RESULT([$supportnesting])
AM_CONDITIONAL(ENABLE_NESTEDOMP, [test "x$supportnesting" = xyes])

# Whether to use pthread library instead of OMP as underlying thread utility.
# Default disabled due to experimental nature.
AC_ARG_ENABLE([pthread], AS_HELP_STRING([--enable-pthread],
      [EXPERIMENTAL: Enable underlying pthread capability instead of OpenMP]))
AS_IF([test "x$enable_pthread" = xyes], [
  AC_MSG_WARN([--enable-pthread WITHIN GPTL IS EXPERIMENTAL: PREFER OPENMP!!!])
  dnl Do the stuff needed for enabling the feature
  AC_MSG_CHECKING([for pthread support])
  AC_CHECK_LIB([pthread], [pthread_mutex_init])
  if test "x$ac_cv_lib_pthread_pthread_mutex_init" = xyes; then
     AC_DEFINE([UNDERLYING_PTHREADS], [1], [use pthreads library for underlying threading])
     AC_MSG_NOTICE([Underlying threading support will be pthread library])
  fi
])
AM_CONDITIONAL(UNDERLYING_PTHREADS, [test "x$ac_cv_lib_pthread_pthread_mutex_init" = xyes])

# Underlying threading must be via openmp (preferred), pthreads, or none at all
if test "x$ac_cv_lib_pthread_pthread_mutex_init" != xyes; then
  if test "x$useomp" = xyes; then
    underlying_omp=yes;
    AC_DEFINE([UNDERLYING_OPENMP], [1], [use openmp for underlying threading])
    AC_MSG_NOTICE([Underlying threading support will be OpenMP library])
  else
    AC_MSG_NOTICE([Underlying threading support is DISABLED: Do not invoke GPTL from threaded regions!])
  fi
fi
AM_CONDITIONAL(UNDERLYING_OPENMP, [test "x$underlying_omp" = xyes])

# Does the Fortran compiler employ double underscores in its name mangling?
# Very few Fortran compilers still do this (e.g. g95 if that's still around)
# Default disabled.
AC_ARG_ENABLE([double-underscore], [AS_HELP_STRING([--enable-double-underscore],
              [use double underscore for Fortran wrappers])])
AS_IF([test "x$enable_double_underscore" = xyes], [
   AC_DEFINE([FORTRANDOUBLEUNDERSCORE], [1],
   [Fortran name mangling uses double underscores e.g. g95])
   AC_MSG_NOTICE([Wrappers for Fortran API will assume double underscoring])
   ],[
   AC_DEFINE([FORTRANUNDERSCORE], [1], [Use single underscore for Fortran wrappers: The usual case])
   AC_MSG_NOTICE([Wrappers for Fortran API will assume single underscoring])
])

# --enable-debug means build with low optimization and add some code inside some #ifdef DEBUG
# Default disabled
use_debug=no
AC_ARG_ENABLE([debug], [AS_HELP_STRING([--enable-debug], [Enable DEBUG ifdef and -g -O0])])
AS_IF([test "x$enable_debug" = xyes], [
  use_debug=yes
  AC_DEFINE([DEBUG], [1], [set debug ifdef])
  CFLAGS="$CFLAGS -g -O0"
  FCFLAGS="$FCFLAGS -g -O0"
])
AM_CONDITIONAL([DEBUG], [test "x$use_debug" = xyes])

# Whether to enable GPTL hooks into the PAPI performance counter library. PAPI must already
# be installed.
# Default disabled.
AC_ARG_ENABLE([papi], [AS_HELP_STRING([--enable-papi],
  [Enable built-in support for papi library. Requires that PAPI is already installed])])
AS_IF([test x$enable_papi = xyes], [
# Check for papi library.
  AC_CHECK_LIB([papi], [PAPI_library_init])
  have_papi=no
  PAPI_PC=""
  if test "x$ac_cv_lib_papi_PAPI_library_init" = xyes; then
    AC_MSG_CHECKING([file system support for PAPI])
    # If we have PAPI library, check /proc/sys/kernel/perf_event_paranoid
    # to see if we have permissions.
    if test -f /proc/sys/kernel/perf_event_paranoid; then
      if test `cat /proc/sys/kernel/perf_event_paranoid` != 1; then
         AC_MSG_ERROR([PAPI library found, but /proc/sys/kernel/perf_event_paranoid != 1
                            try sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'])
      fi
    fi
    AC_DEFINE([HAVE_PAPI], [1], [PAPI library is present and usable])
    have_papi=yes
    PAPI_PC="papi"
    AC_MSG_RESULT($have_papi)
  fi
  # For gptl.pc
  AC_SUBST([PAPI_PC], [$PAPI_PC])
])
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

# Check for existence of /proc, used for obtaining process size.
# Must be disabled when cross-compiling.
# Default enabled.
AC_ARG_ENABLE([slashproc], [AS_HELP_STRING([--disable-slashproc],
    	      [Check for existence of /proc file system: Must be disabled when cross-compiling])])
AS_IF([test "x$enable_slashproc" != xno], [
  AC_CHECK_FILE([/proc],
        [AC_DEFINE([HAVE_SLASHPROC], [1], [/proc exists. Memory checking via /proc enabled])])
])

# We need the math library for some tests.
AC_CHECK_LIB([m], [floor], [], [AC_MSG_ERROR([Can't find or link to the math library])])

# For auto-profiling, libunwind is the default method to recover current function name.
# If disabled or not available, try backtrace. If neither are available, auto-profiling will 
# not be possible. To use auto auto-profiling with GNU or Intel compilers:
# Compile application code with -g and appropriate auto-instrumentation flag
# Probably the appropriate dynamic linking flag (maybe -rdynamic or -Bdynamic) also is needed
# Default enabled
AC_ARG_ENABLE([libunwind],
    AS_HELP_STRING([--disable-libunwind], [Skip check for libunwind--check for backtrace instead],))
found_libunwind=no
AS_IF([test "x$enable_libunwind" != xno], [
  AC_MSG_CHECKING([for libunwind])
  AC_CHECK_LIB(unwind, [unw_backtrace], [found_libunwind=yes], [found_libunwind=no])
  AC_MSG_RESULT([$ac_cv_lib_unwind])
  if test "x$found_libunwind" = xyes; then
    AC_DEFINE([HAVE_LIBUNWIND], [1], [libunwind will be used])
    AC_MSG_NOTICE([libunwind enabled for auto-profiling: -lunwind and dynamic linking flag (see below) possibly also needed])
  fi
])

# Cannot have both unwind and backtrace methods enabled at once
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
AC_CHECK_FUNC([times], [AC_DEFINE([HAVE_TIMES], [1], [times() is available])])

# Check for gettimeofday.
AC_CHECK_FUNC([gettimeofday], [AC_DEFINE([HAVE_GETTIMEOFDAY], [1], [gettimeofday() is available])])

# Do we have MPI?
AC_CHECK_FUNC([MPI_Init], [have_libmpi=yes], [have_libmpi=no])
if test "x$have_libmpi" = xyes; then
# Modify FCFLAGS in case Fortran support is enabled
  FCFLAGS="-DHAVE_LIBMPI $FCFLAGS"
  AC_DEFINE([HAVE_LIBMPI], [1], [Found MPI library])
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
else
  AC_MSG_WARN([MPI_Init not found. GPTL MPI capabilities pr_summary and PMPI will NOT be enabled])
fi
AM_CONDITIONAL([HAVE_LIBMPI], [test "x$have_libmpi" = xyes])

# Did the user specify an MPI launcher other than mpiexec?
AC_MSG_CHECKING([if mpi run command is not mpiexec])
AC_ARG_WITH([mpiexec],
              [AS_HELP_STRING([--with-mpiexec=<command>],
               [Replace mpiexec for launching MPI parallel tests e.g. mpirun, srun])],
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

# Whether to enable auto-profiling with the PMPI profiling layer provided by most MPI distributions
# Default disabled
usepmpi=no
AC_MSG_CHECKING([whether PMPI is to be enabled])
AC_ARG_ENABLE([pmpi], [AS_HELP_STRING([--enable-pmpi], [Build with PMPI support to auto-profile MPI calls])])
AS_IF([test "x$enable_pmpi" = xyes], [
 AC_DEFINE([ENABLE_PMPI], [1], [enable MPI auto-profiling])
 usepmpi=yes
])
AC_MSG_RESULT([$usepmpi])
AM_CONDITIONAL([ENABLE_PMPI], [test x$usepmpi = xyes])

use_cuda=no
AC_MSG_CHECKING([for cuda compiler])
AC_ARG_WITH([nvcc],
              [AS_HELP_STRING([--with-nvcc=<path>],
               [Enable CUDA compilation with nvcc])],
            [NVCC=$with_nvcc],NVCC=no)
AC_MSG_RESULT([$NVCC])
AC_SUBST([NVCC], [$NVCC])

if test "x$NVCC" != xno; then
  use_cuda=yes
  if test "x$use_debug" = xyes; then
    CUFLAGS="$CUFLAGS -lineinfo -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true"
  else
    CUFLAGS="$CUFLAGS           -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true"
  fi
  AC_DEFINE([ENABLE_CUDA], [1], [enable CUDA capability--needed also for OpenACC])
  AC_CONFIG_FILES([cuda/Makefile
                   cuda/include/Makefile
                   cuda/src/Makefile
                   cuda/tests/Makefile
		 ])
fi
AC_SUBST([CUFLAGS], [$CUFLAGS])

# Handle the wart of nvcc-specific behavior
if test "x$use_cuda" = xyes; then
  CPU_SRCDIR=src_nvcc
else
  CPU_SRCDIR=src
fi
AC_SUBST([CPU_SRCDIR], [$CPU_SRCDIR])

# Check if discovering maxwarpid_found is wanted. Computing it adds quite a bit of overhead
# because it is computed in get_warp_num.
# Default disabled due to extra overhead
AC_ARG_ENABLE([maxwarpid_found],
  AS_HELP_STRING([--enable-maxwarpid_found], [Include output of maxwarpid found--adds overhead]))
AS_IF([test "x$enable_maxwarpid_found" = xyes], [
  AC_DEFINE([ENABLE_FOUND], [1], [Include output of maxwarpid found--adds overhead])
])

# Whether Fortran suppport is to be enabled. If so check for working compiler
# Default enabled
fortran_support=no
AC_ARG_ENABLE([fortran], [AS_HELP_STRING([--disable-fortran],
              [Disable fortran support. Only do this if enabling it causes problems])])
AS_IF([test "x$enable_fortran" != "xno"], [
  fortran_support=yes
])
# For some reason AS_IF wrapping the whole thing complains loudly
if test "x$fortran_support" = xyes; then
  AC_LANG_PUSH(Fortran)
  AC_PROG_FC()

# Make sure this file is copied to build directories for tests to work.
  AC_CONFIG_LINKS([fortran/tests/gptlnl:fortran/tests/gptlnl])

# Discover the extension used for Fortran modules (will be stored in $FC_MODEXT)
  AC_FC_MODULE_EXTENSION()

# Test OMP for Fortran and set flags accordingly, only if OMP support was enabled above for C
# AC_OPENMP sets OPENMP_FCFLAGS
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
  AM_CONDITIONAL([ENABLE_CUDA], [test "x$use_cuda" = xyes])

# Modify FCFLAGS for MPI
  if test "x$have_libmpi" = xyes; then
    FCFLAGS="-DHAVE_LIBMPI $FCFLAGS"
  fi

# Modify FCFLAGS for PAPI
  if test "x$have_papi" = xyes; then
    FCFLAGS="-DHAVE_PAPI $FCFLAGS"
  fi
  
# Modify FCFLAGS for CUDA
  if test "x$use_cuda" = xyes; then
    FCFLAGS="-DENABLE_CUDA $FCFLAGS"
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
AM_CONDITIONAL([HAVE_FORTRAN], [test x$fortran_support = "xyes"])

# For gptl.pc
if test "x$fortran_support" = xyes; then
  GPTL_LIBS="-lgptlf -lgptl"
else
  GPTL_LIBS="-lgptl"
fi
if test "x$found_libunwind" = xyes; then
  GPTL_LIBS="$GPTL_LIBS -lunwind"
fi
LDFLAGS_PC="$GPTL_LIBS $LDFLAGS $INSTR_LINK"
# End for gptl.pc

AC_SUBST([LDFLAGS], [$LDFLAGS])
AC_SUBST([LDFLAGS_PC], [$LDFLAGS_PC])
AC_SUBST([OPENMP_CFLAGS], [$OPENMP_CFLAGS])
AC_SUBST([OPENMP_FCFLAGS], [$OPENMP_FCFLAGS])

AC_CONFIG_FILES([tests/run_par_summary_test.sh], [chmod ugo+x tests/run_par_summary_test.sh])
AC_CONFIG_FILES([tests/run_par_pmpi_test.sh], [chmod ugo+x tests/run_par_pmpi_test.sh])
AC_CONFIG_FILES([tests/run_par_global_test.sh], [chmod ugo+x tests/run_par_global_test.sh])
AC_CONFIG_FILES([tests/run_memusage.sh], [chmod ugo+x tests/run_memusage.sh])

# No doxygen--doc is man pages, README, and web pages
# Is doxygen installed?
#AC_CHECK_PROGS([DOXYGEN], [doxygen])
#if test -z "$DOXYGEN"; then
#   AC_MSG_WARN([Doxygen not found - documentation will not be built])
#fi

# Only one of src/Makefile or src_nvcc will be used. Unfortunately AC_CONFIG_FILES doesn't
# take variables or we could just use $CPU_SRCDIR
AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src_nvcc/Makefile
                 gptl.pc
                 include/Makefile
                 tests/Makefile
                 bin/Makefile
		 man/Makefile
	       ])
if test "x$use_cuda" = xyes; then
# OpenACC support REQUIRES cuda support
  AC_ARG_ENABLE([facc],
    AS_HELP_STRING([--enable-facc], [Include Fortran OpenACC support: REQUIRES --enable-cuda]))
  AS_IF([test "x$enable_facc" = xyes], [
    enable_facc=yes
    AC_CONFIG_FILES([facctests/Makefile])
  ])

# OpenACC support REQUIRES cuda support
  AC_ARG_ENABLE([cacc],
      AS_HELP_STRING([--enable-cacc], [Include C/C++ OpenACC support: REQUIRES --enable-cuda]))
    AS_IF([test "x$enable_cacc" = xyes], [
      enable_cacc=yes
      AC_CONFIG_FILES([cacctests/Makefile])
  ])
fi

AM_CONDITIONAL([ENABLE_CUDA], [test "x$use_cuda" = xyes])
AM_CONDITIONAL([ENABLE_FACC], [test "x$enable_facc" = xyes])
AM_CONDITIONAL([ENABLE_CACC], [test "x$enable_cacc" = xyes])

# Build the files listed above.
AC_OUTPUT()
