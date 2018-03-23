CU      = nvcc
CUFLAGS = -g -O0 -G -lineinfo -UDEBUG $(UNDERSCORING) -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true
GPTL    = ..
LDFLAGS = $(CUFLAGS) -L$(GPTL) -lgptl -L$(GPTL)/cuda -lgptl_cuda
OBJS    = getval.o driver_serial.o persist.o

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

all: driver_serial

driver_serial: $(OBJS) ../libgptl.a ../cuda/libgptl_cuda.a
	$(CU) -o $@ $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS) $(OBJS) driver_serial
