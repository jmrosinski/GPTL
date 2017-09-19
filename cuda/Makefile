include ../macros.make

FC      = pgfortran
FFLAGS  = -acc -Minfo=accel -Minfo -ta=tesla:cc60
CU      = nvcc
CUFLAGS = -g -O0 -G -UDEBUG -DENABLE_CUDA $(UNDERSCORING) -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true
ARFLAGS = ruv

OBJS = gptl_device.o util_device.o f_wrappers_device.o gptl_acc.o

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

%.o: %.F90
	$(FC) -c $< $(FFLAGS)

all: $(OBJS)
	$(AR) $(ARFLAGS) lib$(GPULIB).a $(OBJS)
	ranlib lib$(GPULIB).a

gptl.o: gptl_cuda.h private.h
gptl_acc.o: gptl_acc.F90
util.o: private.h

clean:
	$(RM) $(OBJS) *.mod lib$(GPULIB).a
