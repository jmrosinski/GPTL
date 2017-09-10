FC      = mpif90
CC      = mpicc
# ADDING -Mcuda=keepgpu causes runtime failure!
#FFLAGS  = -g -O0 -I$(GPTL) -I$(GPTL)/cuda -acc -Minfo=accel -Minfo -ta=tesla:cc60 -Mcuda=keepgpu
FFLAGS  = -g -O0 -I$(GPTL) -I$(GPTL)/cuda -acc -Minfo=accel -Minfo -ta=tesla:cc60
CFLAGS  = $(FFLAGS)
CU      = nvcc
CUFLAGS = -g -O0 -G -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true
GPTL    = /scratch4/BMC/gsd-hpcs/rosinski/gptl_cuda
LDFLAGS = -L$(GPTL) -lgptl -L$(GPTL)/cuda -lgptl_cuda -L$(CUDALIBDIR) -Mcuda=cuda8 -ta=tesla:cc60
OBJS    = persist.o

%.o: %.F90
	$(FC) -c $< $(FFLAGS)

%.o: %.c
	$(CC) -c $< $(CFLAGS)

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

a.out: $(OBJS) $(GPTL)/libgptl.a $(GPTL)/cuda/libgptl_cuda.a
	$(FC) -o $@ $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS) *.gpu a.out timing.0

