# This automake file generates the Makefile to build the GPTL C library.
# Ed Hartnett 5/18/18

libgptl_la_CPPFLAGS = -I$(top_srcdir)/include 

# This is our output. The GPTL library.
lib_LTLIBRARIES = libgptl.la
libgptl_la_LDFLAGS = -version-info 8:0:0

# These are the source files.
libgptl_la_SOURCES = gptl.c getoverhead.c hashstats.c memstats.c memusage.c util.c

if HAVE_FORTRAN
libgptl_la_SOURCES += f_wrappers.c
endif

if HAVE_PAPI
libgptl_la_SOURCES += gptl_papi.c
endif

if HAVE_LIBMPI
libgptl_la_SOURCES += pr_summary.c
if ENABLE_PMPI
libgptl_la_SOURCES += pmpi.c
if HAVE_FORTRAN
libgptl_la_SOURCES += f_wrappers_pmpi.c
endif
endif
endif

# If inline threading is enabled, the appropriate file will have been #included in gptl.c
if INLINE_THREADING
else
if UNDERLYING_OPENMP
libgptl_la_SOURCES += thread_omp.c
else
if UNDERLYING_PTHREADS
libgptl_la_SOURCES += thread_pthreads.c
else
libgptl_la_SOURCES += thread_none.c
endif
endif
endif
