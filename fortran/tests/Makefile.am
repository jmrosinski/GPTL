## This is an automake file, part of the GPTL-fortran package.

# This is the automake file for GPTL Fortran tests.
# Ed Hartnett, 5/21/18

# Find include directory and src directory (for .mod file).
AM_CPPFLAGS = -I$(top_srcdir)/fortran/src

INSTRFLAG = @INSTRFLAG@
INSTR_LINK = @INSTR_LINK@

# Link to GPTL fortran library.
AM_LDFLAGS = ${top_builddir}/fortran/src/libgptlf.la $(INSTRFLAG)

LDADD = ${top_builddir}/../GPTL/src/libgptl.la
if HAVE_LIBUNWIND
LDADD += -lunwind
endif

MODFILE = ${top_builddir}/fortran/src/gptl.mod

# These programs will be built but not installed.
noinst_PROGRAMS = overhead utrtest handle

# Test programs that will be built for all configurations.
check_PROGRAMS = testbasics errtest nlreader outoforder summary
TESTS          = testbasics errtest nlreader outoforder

# Test dependencies needed.
testbasics_SOURCES = testbasics.F90 ${MODFILE}
errtest_SOURCES    = errtest.F90    ${MODFILE}
nlreader_SOURCES   = nlreader.F90   ${MODFILE}
outoforder_SOURCES = outoforder.F90 ${MODFILE}
summary_SOURCES    = summary.F90    ${MODFILE}

handle_SOURCES     = handle.F90     ${MODFILE}
overhead_SOURCES   = overhead.F90   ${MODFILE}
utrtest_SOURCES    = utrtest.F90    ${MODFILE}

if HAVE_OPENMP
noinst_PROGRAMS  += threadohd
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
TESTS += run_par_summary_test.sh
# Test the PMPI interface (if installed)
if ENABLE_PMPI
check_PROGRAMS += pmpi
TESTS          += run_par_pmpi_test.sh
pmpi_SOURCES = pmpi.F90 ${MODFILE}
endif
else
TESTS += summary
endif

# Test output to be deleted.
CLEANFILES = timing.* *_ohd *.mod

# This file is required for the nlreader test.
EXTRA_DIST = gptlnl
