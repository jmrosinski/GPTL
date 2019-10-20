# This is the automake file for GPTL C tests.
# Ed Hartnett, 5/18/18

# Find include directory.
AM_CPPFLAGS = -I$(top_srcdir)/include

# Need optimization disabled for correct behavior of some tests
CFLAGS += -g -O0 $(OPENMP_CFLAGS)

INSTRFLAG = @INSTRFLAG@
INSTR_LINK = @INSTR_LINK@

# Link to GPTL library.
AM_LDFLAGS = ${top_builddir}/src/libgptl.la

# These programs will be built but not installed.
noinst_PROGRAMS = printwhileon imperfect_nest

# Begin list of programs to be installed
bin_PROGRAMS = gran_overhead

# Test programs that will be built for all configurations.
check_PROGRAMS = tst_simple global
TESTS = tst_simple

if HAVE_INSTRFLAG
TESTS             += cygprofile
noinst_PROGRAMS   += cygprofile
# Hack found online to compile cygprofilesubs.c differently than cygprofile.c: Use a lib
noinst_LIBRARIES   = libcyg.a
cygprofile_LDADD   = libcyg.a
cygprofile_SOURCES = cygprofile.c
libcyg_a_SOURCES   = cygprofilesubs.c
libcyg_a_CFLAGS    = $(INSTRFLAG)
endif

# Test autoprofiling if either HAVE_LIBUNWIND or HAVE_BACKTRACE
if HAVE_LIBUNWIND
  AM_LDFLAGS += $(INSTR_LINK) -lunwind
else
if HAVE_BACKTRACE
  AM_LDFLAGS += $(INSTR_LINK)
endif
endif

if HAVE_OPENMP
noinst_PROGRAMS += omptest
endif

# Build these tests if PAPI is present.
if HAVE_PAPI
check_PROGRAMS += testpapi
TESTS += testpapi
bin_PROGRAMS += gptl_avail knownflopcount
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
TESTS += pmpi run_par_pmpi_test.sh
endif
else
TESTS += global
endif

# Build these if the user selected --enable-nestedomp during configure.
if ENABLE_NESTEDOMP
check_PROGRAMS += nestedomp
TESTS += nestedomp
endif

# Test output to be deleted.
CLEANFILES = timing.*