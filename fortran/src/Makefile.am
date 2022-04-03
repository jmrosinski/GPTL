# Generate the Makefile to build the GPTL-fortran library

# This is our output. The GPTL-fortran library.
if ENABLE_CUDA
lib_LIBRARIES = libgptlf.a
# Look here for the C include files in this dir
libgptlf_a_CPPFLAGS = -I$(top_srcdir)/include
libgptlf_a_FCFLAGS  = -DENABLE_CUDA -I$(top_srcdir)/include
# These are the source files.
libgptlf_a_SOURCES  = gptlf.F90 process_namelist.F90 f_wrappers.c
else
lib_LTLIBRARIES = libgptlf.la
# Look here for the C include files in this dir
libgptlf_la_CPPFLAGS = -I$(top_srcdir)/include
libgptlf_la_FCFLAGS  = -I$(top_srcdir)/include
libgptlf_la_LDFLAGS  = -version-info 0:0:0
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
endif

# Enforce serial build here or bad things happen due to .mod file issues
.NOTPARALLEL:

# Install these in the include directory.
include_HEADERS = gptl.mod 

# Remove module files
CLEANFILES = *.$(FC_MODEXT)
