OBJS = errtest.o papiomptest.o ugex.o utrtest.o summary.o
EXES = errtest   papiomptest   ugex   utrtest   summary

# Don't build these tests for now
ifeq (TRUE,TRUE)
  OBJS += stride.o
  EXES += stride
endif

all: $(EXES)

clean:
	rm -f $(OBJS) $(EXES)

errtest: errtest.o
	$(FC) $(FFLAGS) -o $@ errtest.o $(LDFLAGS)

papiomptest: papiomptest.o
	$(FC) $(FFLAGS) -o $@ papiomptest.o $(LDFLAGS)

stride: stride.o
	$(FC) $(FFLAGS) -o $@ stride.o $(LDFLAGS)

ugex: ugex.o
	$(FC) $(FFLAGS) -o $@ ugex.o $(LDFLAGS)

utrtest: utrtest.o
	$(FC) $(FFLAGS) -o $@ utrtest.o $(LDFLAGS)

summary: summary.o
	$(FC) $(FFLAGS) -o $@ summary.o $(LDFLAGS)
