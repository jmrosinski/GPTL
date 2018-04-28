include ../macros.make
GPTL    = ..
LDFLAGS = $(CUFLAGS) -L$(GPTL) -lgptl -L$(GPTL)/cuda -lgptl_cuda
OBJS    = getval.o driver_serial.o persist.o sleep1.o sleep.o subs.o

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

all: driver_serial

driver_serial: $(OBJS) ../libgptl.a ../cuda/libgptl_cuda.a
	$(CU) -o $@ $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS) $(OBJS) driver_serial
