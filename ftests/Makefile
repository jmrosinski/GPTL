include ../macros.make

# TESTS defines which of EXES have a built-in testing procedure.
TESTS = testbasics nlreader

ifeq ($(HAVE_PAPI),yes)
  TESTS += testinit testpapi
endif
EXES = $(TESTS) errtest utrtest 

ifeq ($(HAVE_MPI),yes)
  EXES += summary
endif

ifeq ($(ENABLE_PMPI),yes)
  EXES += pmpi
endif

ifeq ($(DEBUG),yes)
  ifeq ($(ENABLE_PMPI),yes)
    LIBNAME = gptl_debug_pmpi
  else
    LIBNAME = gptl_debug
  endif
else
  ifeq ($(ENABLE_PMPI),yes)
    LIBNAME = gptl_pmpi
  else
    LIBNAME = gptl
  endif
endif

LDFLAGS = -L.. -l$(LIBNAME) $(ABIFLAGS)
ifeq ($(MPICMD),$(null))
  MPICMD = mpiexec
endif

ifeq ($(OPENMP),yes)
  FFLAGS  += -DTHREADED_OMP $(FOMPFLAG)
  LDFLAGS += $(FOMPFLAG)
else
  ifeq ($(PTHREADS),yes)
# Threaded tests use OpenMP
    FFLAGS  += -DTHREADED_OMP $(FOMPFLAG)
    LDFLAGS += -lpthread $(FOMPFLAG)
  endif
endif

OBJS = $(patsubst %,%.o,$(EXES))

all: $(EXES)

clean:
	$(RM) $(OBJS) $(EXES) timing.*

errtest: errtest.o
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

utrtest: utrtest.o
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

summary: summary.o
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

testinit: testinit.o
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

testpapi: testpapi.o
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

testbasics: testbasics.o
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

nlreader: nlreader.o
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)

pmpi: pmpi.o
	$(FC) $(FFLAGS) -o $@ $< $(LDFLAGS)
pmpi.o: pmpi.F90
	$(FC) -c $(FFLAGS) $<

# Built-in tests to ensure that GPTL is behaving as expected
# Invoke as "make test" from one dir above

test: $(TESTS)
	@$(RM) timing.0
	@echo Running ./testbasics...
	@./testbasics || (echo "Failed" && exit 1)
	@echo Success
	@echo
	@echo Checking contents of timing.0 for a specific region name...
	@grep -q testbasics timing.0 || (echo "Failed" && exit 1)
	@echo Success
	@echo
	@echo Running ./nlreader...
	@./nlreader || (echo "Failed" && exit 1)
	@echo Success
ifeq ($(HAVE_PAPI),yes)
	@echo
	@echo Running ./testinit...
	@./testinit || (echo "Failed" && exit 1)
	@echo
	@echo Running ./testpapi...
	@./testpapi || (echo "Failed" && exit 1)
endif
ifeq ($(ENABLE_PMPI),yes)
	@echo
	@echo Running $(MPICMD) -n 2 ./pmpi
	@env $(MPICMD) -n 2 ./pmpi || (echo "Failed" && exit 1)
endif
	@echo "All Fortran tests passed"
