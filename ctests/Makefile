OBJS = depth.o recursion.o stride.o memusage.o utrtest.o \
       papiomptest.o cygprofile.o cygprofilesubs.o scale.o parentchild.o mpi.o \
       gran_overhead.o load_imbalance.o summary.o

EXES = depth   recursion stride memusage utrtest \
       papiomptest parentchild \
       gran_overhead summary

ifeq ($(HAVE_PAPI),yes)
  OBJS += avail.o rawpapitest.o
  EXES += avail rawpapitest
endif

ifeq ($(HAVE_MPI),yes)
  EXES += scale mpi load_imbalance 
endif

ifeq ($(CC),gcc)
  EXES += cygprofile
  INSTRFLAGS = -finstrument-functions
endif

ifeq ($(CC),pathcc)
  EXES += cygprofile
  INSTRFLAGS = -finstrument-functions
endif

all: $(EXES)

clean:
	rm $(OBJS) $(EXES)

avail: avail.o
	$(CC) -o $@ avail.o $(LDFLAGS)

depth: depth.o
	$(CC) -o $@ depth.o $(LDFLAGS)

overhead: overhead.o
	$(CC) -o $@ overhead.o $(LDFLAGS)

rawpapitest: rawpapitest.o
	$(CC) -o $@ rawpapitest.o $(LDFLAGS)

recursion: recursion.o
	$(CC) -o $@ recursion.o $(LDFLAGS)

exception: exception.o
	$(CC) -o $@ exception.o $(LDFLAGS)

memusage: memusage.o
	$(CC) -o $@ memusage.o $(LDFLAGS)

utrtest: utrtest.o
	$(CC) -o $@ utrtest.o $(LDFLAGS)

papiomptest: papiomptest.o
	$(CC) $(THREADFLAGS) -o $@ papiomptest.c $(LDFLAGS)

cygprofile: cygprofile.o cygprofilesubs.o
	$(CC) -o $@ cygprofile.o cygprofilesubs.o $(LDFLAGS)
cygprofilesubs.o: cygprofilesubs.c
	$(CC) -c $(CFLAGS) $(INSTRFLAGS) -o $@ $<

stride: stride.o
	$(CC) -o $@ stride.o $(LDFLAGS)

scale: scale.o
	$(CC) -o $@ scale.o $(LDFLAGS)

parentchild: parentchild.o
	$(CC) -o $@ parentchild.o $(LDFLAGS)

summary: summary.o
	$(CC) -o $@ summary.o $(LDFLAGS)

mpi: mpi.o
	$(CC) -o $@ mpi.o $(LDFLAGS)

gran_overhead: gran_overhead.o
	$(CC) -o $@ gran_overhead.o $(LDFLAGS)

load_imbalance: load_imbalance.o
	$(CC) -o $@ load_imbalance.o $(LDFLAGS)
