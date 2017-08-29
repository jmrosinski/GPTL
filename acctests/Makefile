include ../macros.make

FC       = pgfortran
CPPFLAGS = -I.. -I../cuda
FFLAGS   = $(CPPFLAGS) -acc -Minfo=accel -Minfo -Mcuda=flushz -ta=tesla:cc60,nofma,keepgpu
CULIBS   = -lcuda -lcudart -lculibos 
LDFLAGS  = -L.. -lgptl -L../cuda -lgptl_cuda -L$(CUDALIBDIR) -lm -ldl -lstdc++ $(CULIBS) -Mcuda=cuda8 -ta=tesla:cc60

SRCS = simple.F90
OBJS = $(patsubst %.F90,%.o,$(SRCS))
EXES = $(patsubst %.F90,%,$(SRCS))

%.o: %.F90
	$(FC) -c $< $(FFLAGS)

all: $(EXES)

simple: simple.o ../libgptl.a ../cuda/libgptl_cuda.a
	$(FC) $< -o $@ $(LDFLAGS)

clean:
	$(RM) $(OBJS) $(EXES)
