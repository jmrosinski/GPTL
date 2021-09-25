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
CUFLAGS += -I$(top_srcdir) -I../../include -I../include --ptxas-options=-v

if ENABLE_FACC
# Install these in the include directory.
include_HEADERS = gptl_acc.$(FC_MODEXT)

libgptl_cuda_a_SOURCES += gptl_acc.F90
# Override FCFLAGS settings because this is OpenACC
FCFLAGS_LOC = -DENABLE_CUDA @FACCFLAGS@
.F90.o:
	$(FC) -c $< $(FCFLAGS_LOC)
# Need this rule to enable parallel make
.F90.mod:
	$(FC) -c $< $(FCFLAGS_LOC)
endif

.cu.o:
	$(NVCC) -c $< $(CUFLAGS)

f_wrappers_device.o: f_wrappers_device.cu ../include/device.h ../include/gptl_cuda.h
gptl_device.o: gptl_device.cu ../include/device.h ../include/gptl_cuda.h
gptl_host.o: gptl_host.cu ../include/device.h
print_gpustats.o: print_gpustats.cu ../include/device.h
util_device.o: util_device.cu ../include/device.h
gptl_acc.o: gptl_acc.F90

CLEANFILES = *.ptx *.ii *.module_id *.cudafe1.* *.fatbin*
