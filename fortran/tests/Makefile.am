## This is an automake file, part of the GPTL-fortran package.

# This is the automake file for GPTL Fortran tests.
# Ed Hartnett, 5/21/18

# Build legacy test with FC
F77=$(FC)

# Find include directory for .inc and src directory for .mod
AM_CPPFLAGS = -I../include -I../src

# For some reason "make distcheck" doesn't populate its Fortran include dir. So we have to
# provide a full path.
#AM_FFLAGS = -I../include
AM_FFLAGS = -I${top_srcdir}/fortran/include

# Link to GPTL fortran and C libraries.
MYLIBS = ${top_builddir}/fortran/src/libgptlf.la ${top_builddir}/src/libgptl.la
AM_LDFLAGS = ${MYLIBS} @INSTR_LINK@

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

MODFILE = ${top_builddir}/fortran/src/gptl.$(FC_MODEXT)

# Enforce serial build here or bad things happen due to .mod file issues
.NOTPARALLEL:

# These programs will be built but not installed.
noinst_PROGRAMS = overhead handle utrtest

# Test programs that will be built for all configurations.
check_PROGRAMS = testbasics errtest nlreader outoforder legacy toomanychars
TESTS          = testbasics errtest nlreader outoforder legacy toomanychars

# Test dependencies needed.
testbasics_SOURCES   = testbasics.F90 ${MODFILE}
errtest_SOURCES      = errtest.F90    ${MODFILE}
nlreader_SOURCES     = nlreader.F90   ${MODFILE}
outoforder_SOURCES   = outoforder.F90 ${MODFILE}
toomanychars_SOURCES = toomanychars.F90 ${MODFILE}
summary_SOURCES      = summary.F90    ${MODFILE}

handle_SOURCES       = handle.F90     ${MODFILE}
overhead_SOURCES     = overhead.F90   ${MODFILE}
legacy_SOURCES       = legacy.F

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
# bin_PROGRAMS will be installed in bin directory
bin_PROGRAMS = print_mpistatus_size
print_mpistatus_size_SOURCES = print_mpistatus_size.F90 
TESTS += run_par_summary_test.sh
# Test the PMPI interface (if installed)
if ENABLE_PMPI
check_PROGRAMS += pmpi inplace
TESTS          += run_par_pmpi_test.sh run_par_inplace_test.sh
pmpi_SOURCES = pmpi.F90 ${MODFILE}
inplace_SOURCES = inplace_subs.F90 inplace.F90 ${MODFILE}
endif
endif

# Test output to be deleted: include ALL possible executables
ALLEXES = overhead handle utrtest testbasics errtest nlreader outoforder legacy omptest \
          threadohd testbacktrace testinit testpapi summary print_mpistatus_size pmpi toomanychars \
          inplace
CLEANFILES = *.$(FC_MODEXT) timing.?????? timing.summary* *_ohd *.mod *.trs *.log *.o $(ALLEXES) \
             *.sh

# This file is required for the nlreader test.
EXTRA_DIST = gptlnl
