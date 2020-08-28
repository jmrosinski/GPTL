# Generate the Makefile to build the GPTL-fortran library

# Look for include files here

# This is our output. The GPTL-fortran library.
if ENABLE_CUDA
lib_LIBRARIES = libgptlf.a
libgptlf_a_FCFLAGS = -DENABLE_CUDA -I$(top_srcdir)/include
libgptlf_a_SOURCES = gptlf.F90 process_namelist.F90
else
lib_LTLIBRARIES = libgptlf.la
libgptlf_la_FCFLAGS = -I$(top_srcdir)/include
libgptlf_la_LDFLAGS = -version-info 0:0:0
libgptlf_la_SOURCES = gptlf.F90 process_namelist.F90
endif

# Follow libtool versioning rules (c:r:a) (current:revision:age)
# lib source changes since last update => increment "r"
# lib API added, removed, or changed since last update => increment "c" and set "r"=0
# API added since last public release => increment "a"
# API removed or changed since last public release => set "a"=0

# Install these in the include directory.
include_HEADERS = gptl.mod

# Remove module files
CLEANFILES = *.$(FC_MODEXT)
