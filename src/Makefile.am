# This automake file generates the Makefile to build the GPTL C library.
# Ed Hartnett 5/18/18

libgptl_la_CPPFLAGS = -I$(top_srcdir)/include 

# This is our output. The GPTL library.
lib_LTLIBRARIES = libgptl.la
libgptl_la_LDFLAGS = -version-info 8:0:0

# These are the source files.
libgptl_la_SOURCES = autoinst.cc getoverhead.cc getter.cc gptl.cc memusage.cc once.cc \
                     postprocess.cc setter.cc util.cc

if HAVE_FORTRAN
libgptl_la_SOURCES += f_wrappers.cc
endif

if HAVE_PAPI
libgptl_la_SOURCES += gptl_papi.cc
endif

if HAVE_LIBMPI
libgptl_la_SOURCES += pr_summary.cc
if ENABLE_PMPI
libgptl_la_SOURCES += pmpi.cc
if HAVE_FORTRAN
libgptl_la_SOURCES += f_wrappers_pmpi.cc
endif
endif
endif

if HAVE_OPENMP
libgptl_la_SOURCES += thread_omp.cc
else
if HAVE_PTHREADS
libgptl_la_SOURCES += thread_pthreads.cc
else
libgptl_la_SOURCES += thread_none.cc
endif
endif
