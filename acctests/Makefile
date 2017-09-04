include ../macros.make

FC       = pgfortran
CC       = pgcc
CPPFLAGS = -I.. -I../cuda
FFLAGS   = -g -O0 $(CPPFLAGS) -acc -Minfo=accel -Minfo -Mcuda=flushz -ta=tesla:cc60,nofma,keepgpu
CFLAGS   = $(FFLAGS)
CULIBS   = -lcuda -lcudart -lculibos 
#LDFLAGS  = -L.. -lgptl -L../cuda -lgptl_cuda -L$(CUDALIBDIR) -lm -ldl -lstdc++ $(CULIBS) -Mcuda=cuda8 -ta=tesla:cc60
LDFLAGS  = -L.. -lgptl -L../cuda -lgptl_cuda -L$(CUDALIBDIR) -lm -ldl -lstdc++ -Mcuda=cuda8 -ta=tesla:cc60
OBJS     = simple.o simplec.o simple2.o

EXES = simple simplec simple2

%.o: %.F90
	$(FC) -c $< $(FFLAGS)

%.o: %.c
	$(CC) -c $< $(CFLAGS)

all: $(EXES)

simple: simple.o ../libgptl.a ../cuda/libgptl_cuda.a
	$(FC) $< -o $@ $(LDFLAGS)

simplec: simplec.o ../libgptl.a ../cuda/libgptl_cuda.a
	$(CC) $< -o $@ $(LDFLAGS)

simple2: simple2.o ../libgptl.a ../cuda/libgptl_cuda.a
	$(FC) $< -o $@ $(LDFLAGS)

clean:
	$(RM) $(OBJS) $(EXES)
