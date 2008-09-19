# This Makefile inherits a number of macros such as HAVE_PAPI from its parent,
# so it's best to invoke it from one dir higher via "make all" or "make test".
# TESTS defines which of EXES have a built-in testing procedure.

TESTS = testbasics

ifeq ($(HAVE_PAPI),yes)
  TESTS += testinit testpapi
endif
EXES = $(TESTS) errtest papiomptest ugex utrtest summary stride

OBJS = $(patsubst %,%.o,$(EXES))

all: $(EXES)

clean:
	$(RM) $(OBJS) $(EXES) timing.*

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

testinit: testinit.o
	$(FC) $(FFLAGS) -o $@ testinit.o $(LDFLAGS)

testpapi: testpapi.o
	$(FC) $(FFLAGS) -o $@ testpapi.o $(LDFLAGS)

testbasics: testbasics.o
	$(FC) $(FFLAGS) -o $@ testbasics.o $(LDFLAGS)

# Built-in tests to ensure that GPTL is behaving as expected
# Invoke as "make test" from one dir above

test: $(TESTS)
	@$(RM) timing.0
	@./testbasics || exit 1
	@echo Checking timing.0...
	@grep -q testbasics timing.0 || exit 2
	@echo Success
ifeq ($(HAVE_PAPI),yes)
	@./testinit || exit 1
	@./testpapi || exit 1
endif
	@echo "All Fortran tests passed"
