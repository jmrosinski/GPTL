# This automake file generates the Makefile to build the GPTL cuda library.

# For now hardwire the compiler
CC = nvcc

# This is our output. The GPTL library of cuda routines
lib_LTLIBRARIES = libgptl_cuda.la

# Look for include files here
libgptl_cuda_la_CPPFLAGS = -I$(top_srcdir)/cuda/include

# Follow libtool versioning rules (c:r:a) (current:revision:age)
# lib source changes since last update => increment "r"
# lib API added, removed, or changed since last update => increment "c" and set "r"=0
# API added since last public release => increment "a"
# API removed or changed since last public release => set "a"=0
libgptl_cuda_la_LDFLAGS = -version-info 0:0:0

# Maximum register count
MAXRREGCOUNT = 8

# These are the source files.
libgptl_cuda_la_SOURCES = gptl_device.cu util_device.cu f_wrappers_device.cu

# For now just hardwire the compile flags
# set max reg count to a small number to minimize GPTL interference with app
CFLAGS = -c -g -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true --maxrregcount 32

.cu.o:
	$(CC) -c $< $(CFLAGS)
