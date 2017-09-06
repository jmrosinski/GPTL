include ../macros.make

FC      = pgfortran
FFLAGS  = -acc -Minfo=accel -Minfo -ta=tesla:cc60,keepgpu
CU      = nvcc
CUFLAGS = -UDEBUG -DENABLE_GPU $(UNDERSCORING) -G -O2 -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true
#CUFLAGS = -UDEBUG -DENABLE_GPU $(UNDERSCORING) -g -O0 -G -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true
#CUFLAGS = -UDEBUG -DENABLE_GPU $(UNDERSCORING) -G -ftz=true -gencode arch=compute_60,code=compute_60
ARFLAGS = ruv

OBJS = gptl.o util.o f_wrappers.o gptlf.o
OBJS += subs.o

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

%.o: %.F90
	$(FC) -c $< $(FFLAGS)

all: $(OBJS)
	$(AR) $(ARFLAGS) lib$(GPULIB).a $(OBJS)
	ranlib lib$(GPULIB).a

gptl.o: gptl.h private.h
gptlf.o: gptlf.F90 gptl.h private.h
util.o: private.h

clean:
	$(RM) $(OBJS) *.mod lib$(GPULIB).a
