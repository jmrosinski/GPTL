CU         = nvcc
# -O2 causes bitwise differences in test codes!
OPTFLAGS   = -g -O2 -UDEBUG
# CUFLAGS only needed in acclib
CUFLAGS    = $(OPTFLAGS) -G -ftz=true -fmad=false -gencode arch=compute_60,code=compute_60 -rdc=true
ARFLAGS    = ruv
LIBNAME    = gptl

OBJS = gptl.o util.o

%.o: %.cu
	$(CU) -c $< $(CUFLAGS)

all: $(OBJS)
	$(AR) $(ARFLAGS) lib$(LIBNAME).a $(OBJS)

gptl.o: gptl.h private.h
util.o: private.h
