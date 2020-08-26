# This automake file generates the Makefile to build the GPTL C library.

# This is our output. The GPTL library.
lib_LIBRARIES = libgptl.a

# Look for include files here
libgptl_a_CPPFLAGS = 

# These are the source files.
libgptl_a_SOURCES = gptl.cu getoverhead.cu hashstats.cu memstats.cu memusage.cu util.cu

if HAVE_FORTRAN
libgptl_a_SOURCES += f_wrappers.cu
endif

if HAVE_PAPI
libgptl_a_SOURCES += gptl_papi.cu
endif

if HAVE_LIBMPI
libgptl_a_SOURCES += pr_summary.cu
if ENABLE_PMPI
libgptl_a_SOURCES += pmpi.cu
if HAVE_FORTRAN
libgptl_a_SOURCES += f_wrappers_pmpi.cu
endif
endif
endif

# If inline threading is enabled, the appropriate file will have been #included in gptl.cu
if INLINE_THREADING
else
if UNDERLYING_OPENMP
libgptl_a_SOURCES += thread_omp.cu
else
if UNDERLYING_PTHREADS
libgptl_a_SOURCES += thread_pthreads.cu
else
libgptl_a_SOURCES += thread_none.cu
endif
endif
endif

all: $(lib_LIBRARIES)

# set max reg count to a small number to minimize GPTL interference with app
libgptl_cuda_a_CFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include -I$(top_srcdir)/cuda/include \
                        -g -O2 $(CUFLAGS) --maxrregcount 32

.cu.o:
	$(CC) -c $< $(libgptl_cuda_a_CFLAGS)

f_wrappers.o: f_wrappers.cu
f_wrappers_pmpi.o: f_wrappers_pmpi.cu
getoverhead.o: getoverhead.cu
gptl.o: gptl.cu
gptl_papi.o: gptl_papi.cu
hashstats.o: hashstats.cu
memstats.o: memstats.cu
memusage.o: memusage.cu
pmpi.o: pmpi.cu
pr_summary.o: pr_summary.cu
thread_none.o: thread_none.cu
thread_omp.o: thread_omp.cu
thread_pthreads.o: thread_pthreads.cu
util.o: util.cu
