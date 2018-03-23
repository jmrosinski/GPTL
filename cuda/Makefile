include ../macros.make

FC      = pgfortran
FFLAGS  = -acc -Minfo=accel -Minfo -ta=tesla:cc60
CU      = nvcc
CUFLAGS = -g -O0 -G -UDEBUG $(UNDERSCORING) -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true
ARFLAGS = ruv

OBJS = gptl_device.o util_device.o f_wrappers_device.o
ifeq ($(ENABLE_ACC),yes)
  OBJS += gptl_acc.o
endif

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

%.o: %.F90
	$(FC) -c $< $(FFLAGS)

all: $(OBJS)
	$(AR) $(ARFLAGS) lib$(GPULIB).a $(OBJS)
	ranlib lib$(GPULIB).a

install: all
	install -m 0644 lib$(GPULIB).a $(INSTALLDIR)/lib
	install -m 0644 gptl_cuda.h $(INSTALLDIR)/include
ifeq ($(ENABLE_ACC),yes)
	install -m 0644 gptl_acc.inc $(INSTALLDIR)/include
	install -m 0644 gptl_acc.mod $(INSTALLDIR)/include
endif

uninstall:
	$(RM) -f $(INSTALLDIR)/lib/lib$(GPULIB)_cuda.a
	$(RM) -f $(INSTALLDIR)/include/gptl_cuda.h
	$(RM) -f $(INSTALLDIR)/include/gptl_acc.inc
	$(RM) -f $(INSTALLDIR)/include/gptl_acc.mod

gptl.o: gptl_cuda.h private.h
gptl_acc.o: gptl_acc.F90
util.o: private.h

clean:
	$(RM) $(OBJS) *.mod lib$(GPULIB).a
