# This automake file generates the Makefile to build the GPTL cuda library.

# This is our output. The GPTL library of cuda routines
lib_LIBRARIES = libgptl_cuda.a

# These are the source files.
libgptl_cuda_a_SOURCES = gptl_host.cu \
	                 gptl_device.cu \
                         util_device.cu \
                         f_wrappers_device.cu \
                         print_gpustats.cu
# set max reg count to a small number to minimize GPTL interference with app
libgptl_cuda_a_CFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include -I$(top_srcdir)/cuda/include \
                        $(CFLAGS) $(CUFLAGS) --maxrregcount 32

if ENABLE_FACC
libgptl_cuda_a_SOURCES += gptl_acc.F90
libgptl_cuda_a_FCFLAGS = -DENABLE_CUDA -acc -ta=tesla:cc60 -m64
.F90.o:
	$(FC) -c $(libgptl_cuda_a_FCFLAGS)
endif

.cu.o:
	$(CC) -c $< $(libgptl_cuda_a_CFLAGS)

f_wrappers_device.o: f_wrappers_device.cu
gptl_device.o: gptl_device.cu
gptl_host.o: gptl_host.cu
print_gpustats.o: print_gpustats.cu
util_device.o: util_device.cu
gptl_acc.o: gptl_acc.F90
