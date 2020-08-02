# Generate the Makefile to build the GPTL-fortran library

# Look for include files here
libgptlf_la_FCFLAGS = -I$(top_srcdir)/include

# This is our output. The GPTL-fortran library.
lib_LTLIBRARIES = libgptlf.la

# Follow libtool versioning rules (c:r:a) (current:revision:age)
# lib source changes since last update => increment "r"
# lib API added, removed, or changed since last update => increment "c" and set "r"=0
# API added since last public release => increment "a"
# API removed or changed since last public release => set "a"=0
libgptlf_la_LDFLAGS = -version-info 0:0:0

# These are the source files.
libgptlf_la_SOURCES = gptlf.F90 process_namelist.F90

if HAVE_PAPI
libgptlf_la_FCFLAGS += -DHAVE_PAPI
endif

if HAVE_LIBMPI
libgptlf_la_FCFLAGS += -DHAVE_LIBMPI
endif

# Each mod file depends on the .o file.
gptl.mod: gptlf.$(OBJEXT)

# Install these in the include directory.
include_HEADERS = gptl.mod

# Remove module files
CLEANFILES = *.$(FC_MODEXT)
