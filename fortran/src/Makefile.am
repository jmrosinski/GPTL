# Generate the Makefile to build the GPTL-fortran library

# Look for include files here
libgptlf_la_FCFLAGS = -I$(top_srcdir)/include
# And here for the C files in this dir
libgptlf_la_CPPFLAGS = -I$(top_srcdir)/include

# This is our output. The GPTL-fortran library.
lib_LTLIBRARIES = libgptlf.la

# Follow libtool versioning rules (c:r:a) (current:revision:age)
# lib source changes since last update => increment "r"
# lib API added, removed, or changed since last update => increment "c" and set "r"=0
# API added since last public release => increment "a"
# API removed or changed since last public release => set "a"=0
libgptlf_la_LDFLAGS = -version-info 0:0:0

# These are the source files.
libgptlf_la_SOURCES = gptlf.F90 process_namelist.F90 f_wrappers.c

if HAVE_PAPI
libgptlf_la_FCFLAGS += -DHAVE_PAPI
endif

if HAVE_LIBMPI
libgptlf_la_FCFLAGS += -DHAVE_LIBMPI
if ENABLE_PMPI
libgptlf_la_SOURCES += f_wrappers_pmpi.c get_f_mpi_in_place.F90
endif
endif

# Enforce serial build here or bad things happen due to .mod file issues
.NOTPARALLEL:

# Install these in the include directory.
include_HEADERS = gptl.mod 

# Remove module files
CLEANFILES = *.$(FC_MODEXT)
