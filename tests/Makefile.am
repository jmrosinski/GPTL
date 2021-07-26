# This is the automake file for GPTL C tests.
# Ed Hartnett, 5/18/18

# Find include directory.
AM_CPPFLAGS = -I$(top_srcdir)/include
if ENABLE_CUDA
AM_CPPFLAGS += -I../cuda/include
endif

# Need optimization disabled for correct behavior of some tests. Guard against multiple "-O0"
# settings mainly because nvcc will barf
if DEBUG
else
CFLAGS += -g -O0
endif

# Link to GPTL library. CUDA lib also required when enabled due to static build
# -cudalib and -ta args also required to avoid unsat externals
AM_LDFLAGS = -L../src -lgptl
if ENABLE_CUDA
  AM_LDFLAGS += -L../cuda/src -lgptl_cuda -cudalib=nvshmem -ta=tesla,cc61
endif
AM_LDFLAGS += @INSTR_LINK@

# These programs will be built but not installed.
noinst_PROGRAMS = printwhileon imperfect_nest

# bin_PROGRAMS will be installed in bin directory on "make install"
bin_PROGRAMS = gran_overhead

# Test programs that will be built for all configurations.
# memusage test requires a script because the output needs to be examined
check_PROGRAMS = tst_simple global badhandle memusage
TESTS = tst_simple badhandle run_memusage.sh
noinst_PROGRAMS += memusage

if HAVE_INSTRFLAG
if HAVE_LIBUNWIND
TESTS             += cygprofile
noinst_PROGRAMS   += cygprofile
else
if HAVE_BACKTRACE
TESTS             += cygprofile
noinst_PROGRAMS   += cygprofile
endif
endif

# Hack found online to compile cygprofilesubs.c differently than cygprofile.c: Use a lib
noinst_LIBRARIES   = libcyg.a
cygprofile_LDADD   = libcyg.a
cygprofile_SOURCES = cygprofile.c
libcyg_a_SOURCES   = cygprofilesubs.c
libcyg_a_CFLAGS    = @INSTRFLAG@
endif

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

if HAVE_OPENMP
check_PROGRAMS += omptest
TESTS += omptest
endif

# Build these tests if PAPI is present.
if HAVE_PAPI
check_PROGRAMS += testpapi
TESTS += testpapi
bin_PROGRAMS += gptl_avail
if HAVE_OPENMP
noinst_PROGRAMS += papiomptest
endif
endif

# Build these tests if MPI is present.
if HAVE_LIBMPI
noinst_PROGRAMS += global
TESTS += run_par_global_test.sh

check_PROGRAMS += summary
noinst_PROGRAMS += summary
TESTS += run_par_summary_test.sh
if ENABLE_PMPI
check_PROGRAMS += pmpi
noinst_PROGRAMS += pmpi
TESTS += run_par_pmpi_test.sh
endif
endif

# Build these if the user selected --enable-nestedomp during configure.
if ENABLE_NESTEDOMP
check_PROGRAMS += nestedomp
TESTS += nestedomp
endif

# Test output to be deleted: include ALL possible executables
ALLEXES = printwhileon imperfect_nest gran_overhead tst_simple global cygprofile omptest \
          testpapi gptl_avail knownflopcount papiomptest summary pmpi nestedomp badhandle \
          memusage
CLEANFILES = timing.?????? timing.summary* *.trs *.log *.o out.memusage $(ALLEXES)
