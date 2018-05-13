CU = /opt/pgi/linux86-64/2018/cuda/8.0/bin/nvcc
CUFLAGS = -g -O0 -G -lineinfo -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true
LDFLAGS = $(CUFLAGS) -lstdc++
OBJS    = main.o sim.o

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

gptlsim: $(OBJS)
	$(CU) -o $@ $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS) gptlsim
