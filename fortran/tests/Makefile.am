## This is an automake file, part of the GPTL-fortran package.

# This is the automake file for GPTL Fortran tests.
# Ed Hartnett, 5/21/18

# Find include directory and src directory (for .mod file).
F77=$(FC)
AM_CPPFLAGS = -I../include -I../src

# Link to GPTL fortran library.
AM_LDFLAGS = ${top_builddir}/fortran/src/libgptlf.la ${top_builddir}/src/libgptl.la @INSTR_LINK@

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

MODFILE = ${top_builddir}/fortran/src/gptl.mod

# These programs will be built but not installed.
noinst_PROGRAMS = overhead handle
bin_PROGRAMS = utrtest

# Test programs that will be built for all configurations.
check_PROGRAMS = testbasics errtest nlreader outoforder legacy
TESTS          = testbasics errtest nlreader outoforder legacy

# Test dependencies needed.
testbasics_SOURCES = testbasics.F90 ${MODFILE}
errtest_SOURCES    = errtest.F90    ${MODFILE}
nlreader_SOURCES   = nlreader.F90   ${MODFILE}
outoforder_SOURCES = outoforder.F90 ${MODFILE}
summary_SOURCES    = summary.F90    ${MODFILE}

handle_SOURCES     = handle.F90     ${MODFILE}
overhead_SOURCES   = overhead.F90   ${MODFILE}
legacy_SOURCES     = legacy.F

# Hack found online to compile input files differently: Use a lib
noinst_LIBRARIES      = libprofsubs.a
utrtest_LDADD         = libprofsubs.a
utrtest_SOURCES       = utrtest.F90 ${MODFILE}
libprofsubs_a_SOURCES = autoprofsubs.F90
libprofsubs_a_FCFLAGS = @INSTRFLAG@

if HAVE_FORT_OPENMP
check_PROGRAMS += omptest
TESTS += omptest
omptest_SOURCES = omptest.F90 ${MODFILE}

noinst_PROGRAMS += threadohd
threadohd_SOURCES = threadohd.F90 ${MODFILE}
endif

# Extra test if we have backtrace.
if HAVE_BACKTRACE
check_PROGRAMS       += testbacktrace
TESTS                += testbacktrace
testbacktrace_SOURCES = testbacktrace.F90 ${MODFILE}
endif

# Test PAPI functionality if libpapi was found.
if HAVE_PAPI
AM_CPPFLAGS     += -DHAVE_PAPI
check_PROGRAMS  += testinit testpapi
TESTS           += testinit testpapi 
testpapi_SOURCES = testpapi.F90 ${MODFILE}
testinit_SOURCES = testinit.F90 ${MODFILE}
endif

# MPI program tests
if HAVE_LIBMPI
check_PROGRAMS += summary
bin_PROGRAMS += print_mpistatus_size
print_mpistatus_size_SOURCES = print_mpistatus_size.F90 
TESTS += run_par_summary_test.sh
# Test the PMPI interface (if installed)
if ENABLE_PMPI
check_PROGRAMS += pmpi
TESTS          += run_par_pmpi_test.sh
pmpi_SOURCES = pmpi.F90 ${MODFILE}
endif
endif

# Test output to be deleted: include ALL possible executables
ALLEXES = overhead handle utrtest testbasics errtest nlreader outoforder legacy omptest \
          threadohd testbacktrace testinit testpapi summary print_mpistatus_size pmpi
CLEANFILES = timing.?????? *_ohd *.mod *.trs *.log *.o $(ALLEXES)

# This file is required for the nlreader test.
EXTRA_DIST = gptlnl
