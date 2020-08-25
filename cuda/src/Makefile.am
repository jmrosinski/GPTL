# This automake file generates the Makefile to build the GPTL cuda library.

# For now hardwire the compiler
NVCC = nvcc

# Look for include files here
libgptl_cudaincludedir = ../include

# This is our output. The GPTL library of cuda routines
lib_LIBRARIES = libgptl_cuda.a

# These are the source files.
libgptl_cuda_a_SOURCES = gptl_host.cu \
	                 gptl_device.cu \
                         util_device.cu \
                         f_wrappers_device.cu \
                         print_gpustats.cu

# For now just hardwire the compile flags
# set max reg count to a small number to minimize GPTL interference with app
libgptl_cuda_a_CFLAGS = -g -O2 -I$(top_srcdir) -I$(top_srcdir)/include -I$(top_srcdir)/cuda/include \
                        -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true \
                        --maxrregcount 32

libgptl_cuda_a_AR = $(NVCC) $(libgptl_cuda_a_CFLAGS) -lib -o

.cu.o:
	$(NVCC) -c $< $(libgptl_cuda_a_CFLAGS)

f_wrappers_device.o: f_wrappers_device.cu
gptl_device.o: gptl_device.cu
gptl_host.o: gptl_host.cu
print_gpustats.o: print_gpustats.cu
util_device.o: util_device.cu

