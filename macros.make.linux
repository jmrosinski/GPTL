# Where to install GPTL library, include files, and man pages
INSTALLDIR = /usr/local

# C compiler and flags: symbols, high optimization, inlining
CC     = gcc
CFLAGS = -g -O3 -finline-functions -Winline -Wall

# Set ABI flags for non-default ABIs (e.g. 64-bit addressing)
ABIFLAGS =

# To enable OpenMP threading, set OPENMP=yes and define the compiler flag. Otherwise,
# set OPENMP=no. OpenMP applications linked with GPTL as built with OPENMP=no will NOT
# be thread-safe
OPENMP = yes
ifeq ($(OPENMP),yes)
  COMPFLAG = -fopenmp
else
# OpenMP threading not enabled: threading may be enabled via the pthreads library.
# If so, add -DTHREADED_PTHREADS to CFLAGS. -lpthread will probably be needed on LDFLAGS.
  CFLAGS  += -DTHREADED_PTHREADS
  LDFLAGS += -lpthread
endif

# For gcc, -Dinline=inline is a no-op. For other C compilers, things like -Dinline=__inline__
# may be required. Autoconf test AC_C_INLINE can find the right definition.
INLINEFLAG  = -Dinline=inline

# To build the Fortran interface, set FORTRAN=yes and define the entries under
# ifeq ($(FORTRAN),yes). Otherwise, set FORTRAN=no and skip this section.
FORTRAN = yes
ifeq ($(FORTRAN),yes)
# Fortran name mangling: possibilities are: leave UNDERSCORING blank (none),
# -DFORTRANDOUBLEUNDERSCORE (e.g. g77), and -DFORTRANUNDERSCORE (e.g. gfortran)
#  UNDERSCORING =
#  UNDERSCORING = -DFORTRANDOUBLEUNDERSCORE
  UNDERSCORING = -DFORTRANUNDERSCORE

# Set Fortran compiler, flags, and (if OpenMP enabled) OpenMP compiler flag
# These settings are only used for the Fortran test applications.
  FC     = gfortran
  FFLAGS = -g -O2
  ifeq ($(OPENMP),yes)
    FOMPFLAG = -fopenmp
  endif
endif

# Archiver: normally it's just ar
AR = ar

# PAPI: To enable, set HAVE_PAPI=yes. Then set inc and lib info if needed.
# PAPI_LIBNAME: Test applications will be linked with -l$(PAPI_LIBNAME)
HAVE_PAPI = yes
ifeq ($(HAVE_PAPI),yes)
  PAPI_INCDIR  = /usr/local/include
  PAPI_LIBDIR  = /usr/local/lib
  PAPI_LIBNAME = papi
endif

# MPI: To enable, set HAVE_MPI=yes. Then set inc and lib info if needed.
# If CC=mpicc for example, MPI_INCDIR MPI_LIBDIR and MPI_LIBNAME can all be blank.
# If MPI_LIBNAME is set, test applications will be linked with -l$(MPI_LIBNAME)
HAVE_MPI    = yes
ifeq ($(HAVE_MPI),yes)
  MPI_INCDIR  = /usr/local/include
  MPI_LIBDIR  = /usr/local/lib
  MPI_LIBNAME = mpich
endif

# librt.a is an option for gathering wallclock time stats on some machines. Setting
# HAVE_LIBRT=yes enables this, but will probably require linking with -lrt
HAVE_LIBRT = yes

# Only define HAVE_NANOTIME if this is a x86
# If HAVE_NANOTIME=yes, set BIT64=yes if this is an x86_64
HAVE_NANOTIME = yes
ifeq ($(HAVE_NANOTIME),yes)
  BIT64 = no
endif

# Some old compilers don't support vprintf. Set to "no" in this case
HAVE_VPRINTF = yes

# Some old compilers don't support the C times() function. Set to "no" in this case
HAVE_TIMES = yes

# gettimeofday() should be available everywhere. But if not, set to "no"
HAVE_GETTIMEOFDAY = yes
