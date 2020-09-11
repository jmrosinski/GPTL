# This automake file generates the Makefile to build the GPTL cuda library.

# This is our output. The GPTL library of cuda routines
lib_LIBRARIES = libgptl_cuda.a

# These are the source files.
libgptl_cuda_a_SOURCES = gptl_host.cu \
	                 gptl_device.cu \
                         util_device.cu \
                         f_wrappers_device.cu \
                         print_gpustats.cu

# Prepend include dirs to CUFLAGS, and set max reg count to a small number
# to minimize GPTL interference with app
CUFLAGS += -I$(top_srcdir) -I../../include -I../include --maxrregcount 32

if ENABLE_FACC
# Install these in the include directory.
include_HEADERS = gptl_acc.$(FC_MODEXT)

libgptl_cuda_a_SOURCES += gptl_acc.F90
# Override FCFLAGS settings because this is OpenACC
FCFLAGS_LOC = -DENABLE_CUDA @FACCFLAGS@
.F90.o:
	$(FC) -c $< $(FCFLAGS_LOC)
endif

.cu.o:
	$(NVCC) -c $< $(CUFLAGS)

f_wrappers_device.o: f_wrappers_device.cu
gptl_device.o: gptl_device.cu
gptl_host.o: gptl_host.cu
print_gpustats.o: print_gpustats.cu
util_device.o: util_device.cu
gptl_acc.o: gptl_acc.F90
