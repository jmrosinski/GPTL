include ../macros.make
GPTL    = ..
LDFLAGS = $(CUFLAGS) -L$(GPTL) -lgptl -L$(GPTL)/cuda -lgptl_cuda -lstdc++
OBJS    = getval.o driver_serial.o persist.o sleep1.o sleep.o subs.o

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

all: driver_serial test1

driver_serial: $(OBJS) ../libgptl.a ../cuda/libgptl_cuda.a
	$(CU) -o $@ $(OBJS) $(LDFLAGS)

test1: test1.cu ../libgptl.a ../cuda/libgptl_cuda.a
	$(CU) -o $@ test1.cu $(LDFLAGS)

INTERMED_FILES = *.ii *.ptx *.cudafe1.* *.fatbin* *.module_id *_dlink.*
clean:
	$(RM) $(OBJS) driver_serial $(INTERMED_FILES)
