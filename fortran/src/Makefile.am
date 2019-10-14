# This is part of NOAAs GPTL-fortran package.

# This automake file generates the Makefile to build the GPTL-fotran
# library.
# Ed Hartnett 5/21/18

libgptlf_la_FCFLAGS = -I$(top_srcdir)/include

if HAVE_PAPI
libgptlf_la_FCFLAGS += -DHAVE_PAPI
endif

if HAVE_LIBMPI
libgptlf_la_FCFLAGS += -DHAVE_LIBMPI
endif

# This is our output. The GPTL-fortran library.
lib_LTLIBRARIES = libgptlf.la
libgptlf_la_SOURCES = gptlf.F90 printmpistatussize.F90	\
process_namelist.F90

# Each mod file depends on the .o file.
gptl.mod: gptlf.$(OBJEXT)

# Install these in the include directory.
include_HEADERS = gptl.mod

CLEANFILES = *.mod
