# This Makefile inherits a number of macros such as HAVE_MPI from its parent,
# so it's best to invoke it from one dir higher via "make all" or "make test".

# TESTS defines which EXES to run when "make test" is done.
# Always run "summary". Add others depending on settings.
TESTS =
ifeq ($(HAVE_MPI),yes)
  TESTS += summary
endif

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

EXES = $(TESTS) papiomptest gptl_gran_overhead

UTILEXES = gptl_gran_overhead
ifeq ($(HAVE_PAPI),yes)
  UTILEXES += gptl_avail gptl_knownflopcount
endif
EXES += $(UTILEXES)

OBJS += $(patsubst %,%.o,$(EXES))

all: $(EXES)

clean:
	$(RM) *.o $(EXES) timing.*

gptl_avail: avail.o
	$(CC) -o $@ $< $(LDFLAGS)

gptl_knownflopcount: knownflopcount.o
	$(CC) -o $@ $< $(LDFLAGS)

gptl_gran_overhead: gran_overhead.o
	$(CC) -o $@ $< $(LDFLAGS)

papiomptest: papiomptest.o
	$(CC) $(THREADFLAGS) -o $@ $< $(LDFLAGS)

cygprofile: cygprofile.o cygprofilesubs.o
	$(CC) -o $@ cygprofile.o cygprofilesubs.o $(LDFLAGS) $(INSTRFLAG)
cygprofilesubs.o: cygprofilesubs.c
	$(CC) -c $(CFLAGS) $(INSTRFLAG) -o $@ $<

profcxx: profcxx.o
	$(CXX) -o $@ $< $(LDFLAGS) $(INSTRFLAG)
profcxx.o: profcxx.C myclasses.h
	$(CXX) -c $(CFLAGS) $(INSTRFLAG) -o $@ $<

summary: summary.o
	$(CC) -o $@ $< $(LDFLAGS)

testpapi: testpapi.o
	$(CC) -o $@ $< $(LDFLAGS)

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

install: $(UTILEXES)
ifneq ($(UTILEXES),)
	install -m 0755 $(UTILEXES) $(INSTALLDIR)/bin && exit 0
else
	@exit 0
endif
