# This Makefile inherits a number of macros such as HAVE_MPI from its parent,
# so it's best to invoke it from one dir higher via "make all" or "make test".

# TESTS defines which EXES to run when "make test" is done.
# Always run "summary". Add others depending on settings.
TESTS = summary
ifeq ($(HAVE_PAPI),yes)
  TESTS += testpapi
endif

# Auto-profiling 
ifeq ($(TEST_AUTOPROFILE),yes)
  TESTS += cygprofile profcxx
# Initialize OBJS: cygprofilesubs.o is the one .o file needed that patsubst can't
# set automatically
  OBJS = cygprofilesubs.o
endif

# EXES is TESTS plus some other sample codes.
# Items in EXES but not TESTS are built by "make all" but
# not run by "make test"

EXES = $(TESTS) depth recursion stride memusage utrtest \
       papiomptest parentchild gran_overhead rawpapitest

ifeq ($(HAVE_PAPI),yes)
  EXES += avail papiacc
endif

ifeq ($(HAVE_MPI),yes)
  EXES += scale mpi load_imbalance 
endif

OBJS += $(patsubst %,%.o,$(EXES))

all: $(EXES)

clean:
	$(RM) $(OBJS) $(EXES) timing.*

avail: avail.o
	$(CC) -o $@ $< $(LDFLAGS)

depth: depth.o
	$(CC) -o $@ $< $(LDFLAGS)

overhead: overhead.o
	$(CC) -o $@ $< $(LDFLAGS)

rawpapitest: rawpapitest.o
	$(CC) -o $@ $< $(LDFLAGS)

recursion: recursion.o
	$(CC) -o $@ $< $(LDFLAGS)

exception: exception.o
	$(CC) -o $@ $< $(LDFLAGS)

memusage: memusage.o
	$(CC) -o $@ $< $(LDFLAGS)

utrtest: utrtest.o
	$(CC) -o $@ $< $(LDFLAGS)

papiomptest: papiomptest.o
	$(CC) $(THREADFLAGS) -o $@ $< $(LDFLAGS)

cygprofile: cygprofile.o cygprofilesubs.o
	$(CC) -o $@ cygprofile.o cygprofilesubs.o $(LDFLAGS) $(INSTRFLAG)
cygprofilesubs.o: cygprofilesubs.c
	$(CC) -c $(CFLAGS) $(INSTRFLAG) -o $@ $<

profcxx: profcxx.o
	$(CXX) -o $@ $< $(LDFLAGS) $(INSTRFLAG)
profcxx.o: profcxx.C
	$(CXX) -c $(CFLAGS) $(INSTRFLAG) -o $@ $<

stride: stride.o
	$(CC) -o $@ $< $(LDFLAGS)

scale: scale.o
	$(CC) -o $@ $< $(LDFLAGS)

parentchild: parentchild.o
	$(CC) -o $@ $< $(LDFLAGS)

summary: summary.o
	$(CC) -o $@ $< $(LDFLAGS)

mpi: mpi.o
	$(CC) -o $@ $< $(LDFLAGS)

gran_overhead: gran_overhead.o
	$(CC) -o $@ $< $(LDFLAGS)

load_imbalance: load_imbalance.o
	$(CC) -o $@ $< $(LDFLAGS)

testpapi: testpapi.o
	$(CC) -o $@ $< $(LDFLAGS)

toomanycounters: toomanycounters.o
	$(CC) -o $@ $< $(LDFLAGS)

#Optimize papiacc to get good estimate of GPTL_CI
papiacc: papiacc.o
	$(CC) -O -c -o papiacc.o papiacc.c $(CFLAGS)
	$(CC) -o $@ papiacc.o $(LDFLAGS)

# Built-in tests to ensure that GPTL is behaving as expected
# Invoke as "make test" from one dir above

test: $(TESTS)
	@echo

ifeq ($(HAVE_PAPI),yes)
	@./testpapi        || exit 1
	@echo
endif

ifeq ($(HAVE_MPI),yes)
	@echo Running env OMP_NUM_THREADS=2 $(MPICMD) -n 2 ./summary...
	@env OMP_NUM_THREADS=2 $(MPICMD) -n 2 ./summary || (echo "Failure to run" && exit 1)
	@echo "Shell is checking contents of timing.summary produced by ./summary for a specific region name..."
	@grep -q sleep timing.summary && echo "Success" || (echo "Failure" && exit 1)
	@echo
endif

ifeq ($(TEST_AUTOPROFILE),yes)
	@echo "Testing auto-profiling..."
	@./cygprofile || (echo "Failure to run" && exit 1)
	@(../hex2name.pl ./cygprofile timing.0 | grep -q callutil100times) && echo "Success" || (echo "Failure" && exit 1)
	@echo
endif
	@echo "All C tests passed"
