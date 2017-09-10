include ../macros.make

FC      = pgfortran
FFLAGS  = -acc -Minfo=accel -Minfo -ta=tesla:cc60
CU      = nvcc
CUFLAGS = -UDEBUG -DENABLE_ACC $(UNDERSCORING) -G -O0 -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true
#CUFLAGS = -UDEBUG -DENABLE_ACC $(UNDERSCORING) -g -O0 -G -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true
#CUFLAGS = -UDEBUG -DENABLE_ACC $(UNDERSCORING) -G -ftz=true -gencode arch=compute_60,code=compute_60
ARFLAGS = ruv

OBJS = gptl.o util.o f_wrappers.o gptl_acc.o

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

%.o: %.F90
	$(FC) -c $< $(FFLAGS)

all: $(OBJS)
	$(AR) $(ARFLAGS) lib$(GPULIB).a $(OBJS)
	ranlib lib$(GPULIB).a

gptl.o: gptl_acc.h private.h
gptl_acc.o: gptl_acc.F90
util.o: private.h

clean:
	$(RM) $(OBJS) *.mod lib$(GPULIB).a
