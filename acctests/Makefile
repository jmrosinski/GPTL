FC   = mpif90
GPTL = ..
ifeq ($(FLAGS),fail)
  FFLAGS  = -g -O0 -I$(GPTL) -I$(GPTL)/cuda -acc -Minfo=accel -Minfo -Mcuda=cuda8 -ta=tesla:cc60
  LDFLAGS = -L$(GPTL) -lgptl -L$(GPTL)/cuda -lgptl_cuda -Mcuda=cuda8,rdc -ta=tesla:cc60
else
  FFLAGS  = -g -O0 -I$(GPTL) -I$(GPTL)/cuda -acc -Minfo=accel -Minfo              -ta=tesla:cc60
  LDFLAGS = -L$(GPTL) -lgptl -L$(GPTL)/cuda -lgptl_cuda -Mcuda=cuda8     -ta=tesla:cc60
endif
OBJS1 = persist.o
OBJS2 = persist_mpi.o
OBJS = $(OBJS1) $(OBJS2)

%.o: %.F90
	$(FC) -c $< $(FFLAGS)

all: persist persist_mpi

persist: $(OBJS1) ../libgptl.a ../cuda/libgptl_cuda.a
	$(FC) -o $@ $(OBJS1) $(LDFLAGS)

persist_mpi: $(OBJS2) ../libgptl.a ../cuda/libgptl_cuda.a
	$(FC) -o $@ $(OBJS2) $(LDFLAGS)

clean:
	$(RM) $(OBJS) *.gpu persist persist_mpi

