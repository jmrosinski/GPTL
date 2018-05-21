## This is an automake file, part of the GPTL package.

# This is the automake file for GPTL C tests.
# Ed Hartnett, 5/18/18

# Find include directory.
AM_CPPFLAGS = -I$(top_srcdir)/include

# Link to GPTL library.
AM_LDFLAGS = ${top_builddir}/src/clib/libgptl.la

# These programs will be built but not installed.
noinst_PROGRAMS = gran_overhead printwhileon

# Test programs that will be built for all configurations.
check_PROGRAMS = tst_simple global
TESTS = tst_simple global

# Build these tests if PAPI is present.
if HAVE_PAPI
check_PROGRAMS += avail testpapi
TESTS += avail testpapi
noinst_PROGRAMS += papiomptest knownflopcount
endif

# Build these tests if MPI is present.
if HAVE_MPI
check_PROGRAMS += summary pmpi
TESTS += run_par_test.sh
endif

# Build these if the user selected --enable-autoprofile-tests during configure.
if TEST_AUTOPROFILE
check_PROGRAMS += cygprofile cygprofilesubs
TESTS += cygprofile cygprofilesubs
endif

# Build these if the user selected --enable-nestedomp during configure.
if ENABLE_NESTEDOMP
check_PROGRAMS += nestedomp
TESTS += nestedomp
endif

# Test output to be deleted.
CLEANFILES = timing.*
