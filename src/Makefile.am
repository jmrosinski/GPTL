# This automake file generates the Makefile to build the GPTL C library.

# Look for include files here
libgptl_la_CPPFLAGS = -I$(top_srcdir)/include 

# This is our output. The GPTL library.
lib_LTLIBRARIES = libgptl.la

# Follow libtool versioning rules (c:r:a) (current:revision:age)
# lib source changes since last update => increment "r"
# lib API added, removed, or changed since last update => increment "c" and set "r"=0
# API added since last public release => increment "a"
# API removed or changed since last public release => set "a"=0
libgptl_la_LDFLAGS = -version-info 0:0:0

# These are the source files.
libgptl_la_SOURCES = autoinst.cc overhead.cc getter.cc gptl.cc hashstats.cc memstats.cc \
                     memusage.cc once.cc postprocess.cc setter.cc util.cc

if HAVE_PAPI
libgptl_la_SOURCES += gptl_papi.cc
endif

if HAVE_LIBMPI
libgptl_la_SOURCES += pr_summary.cc
if ENABLE_PMPI
libgptl_la_SOURCES += pmpi.cc
endif
endif

# If inline threading is enabled, the appropriate file will have been #included in gptl.c
if INLINE_THREADING
else
if UNDERLYING_OPENMP
libgptl_la_SOURCES += thread_omp.cc
else
if UNDERLYING_PTHREADS
libgptl_la_SOURCES += thread_pthreads.cc
else
libgptl_la_SOURCES += thread_none.cc
endif
endif
endif
