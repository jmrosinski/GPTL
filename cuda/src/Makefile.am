# This automake file generates the Makefile to build the GPTL cuda library.

# This is our output. The GPTL library of cuda routines
lib_LIBRARIES = libgptl_cuda.a

# These are the source files.
libgptl_cuda_a_SOURCES = api.cu \
                         f_wrappers_device.cu \
                         gptl_host.cu \
                         gpustats.cu \
                         init_final.cu \
                         output.cu \
                         overhead.cu \
                         stringfuncs.cu \
                         util.cu

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

api.o: api.cu ../include/gptl_cuda.h ../include/api.h ../include/stringfuncs.h \
       ../include/util.h ../include/timingohd.h
f_wrappers_device.o: f_wrappers_device.cu ../include/gptl_cuda.h ../../include/devicehost.h
gptl_acc.o: gptl_acc.F90
gptl_host.o: gptl_host.cu ../include/api.h
gpustats.o: gpustats.cu ../include/gpustats.h ../include/api.h ../include/stringfuncs.h \
            ../include/util.h ../include/output.h
init_final.o: init_final.cu ../include/api.h ../include/util.h ../include/timingohd.h
output.o: output.cu ../include/output.h ../include/api.h ../include/util.h ../include/overhead.h \
          ../include/timingohd.h
overhead.o: overhead.cu ../include/overhead.h ../../include/private.h ../include/api.h \
            ../include/stringfuncs.h
stringfuncs.o: stringfuncs.cu stringfuncs.h
util.o: util.cu ../include/util.h ../include/api.h ../include/output.h

CLEANFILES = *.ptx *.ii *.module_id *.cudafe1.* *.fatbin*
