FC   = mpif90
GPTL = ..
ifeq ($(FLAGS),fail)
  FFLAGS  = -g -O0 -I$(GPTL) -I$(GPTL)/cuda -acc -Minfo=accel -Minfo -Mcuda=cuda8 -ta=tesla:cc60
  LDFLAGS = -L$(GPTL) -lgptl -L$(GPTL)/cuda -lgptl_cuda -Mcuda=cuda8,rdc -ta=tesla:cc60
else
  FFLAGS  = -g -O0 -I$(GPTL) -I$(GPTL)/cuda -acc -Minfo=accel -Minfo              -ta=tesla:cc60,keepgpu
  LDFLAGS = -L$(GPTL) -lgptl -L$(GPTL)/cuda -lgptl_cuda -Mcuda=cuda8     -ta=tesla:cc60
endif

# DO NOT ADD THIS FLAG: IT SCREWS UP THE OUTPUT!!!
#FFLAGS += -Mcuda

COMMONOBJS = persist.o getval.o
MPROG1 = driver_serial.o
MPROG2 = driver_mpi.o
OBJS1 = $(MPROG1) $(COMMONOBJS)
OBJS2 = $(MPROG2)  $(COMMONOBJS)

%.o: %.F90
	$(FC) -c $< $(FFLAGS)

all: driver_serial driver_mpi

driver_serial: $(OBJS1) ../libgptl.a ../cuda/libgptl_cuda.a
	$(FC) -o $@ $(OBJS1) $(LDFLAGS)

driver_mpi: $(OBJS2) ../libgptl.a ../cuda/libgptl_cuda.a
	$(FC) -o $@ $(OBJS2) $(LDFLAGS)

clean:
	$(RM) $(MPROG1) $(MPROG2) $(OBJS1) $(OBJS2) *.mod *.gpu driver_serial driver_mpi

$(MPROG1): getval.o
$(MPROG2): getval.o
