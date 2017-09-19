include macros.make

ifeq ($(findstring xlf, $(FC)),xlf)
  DEFINE = -WF,-D
else
  DEFINE = -D
endif

null =
OBJS = gptl.o util.o memusage.o getoverhead.o \
       hashstats.o memstats.o pr_summary.o print_rusage.o

LIBNAME = gptl

# Only build/run acctests if ENABLE_ACC is set
ifeq ($(ENABLE_CUDA),yes)
  MAKETESTS = acctests/all
  RUNTESTS = acctests/test
else
  MAKETESTS = ctests/all
  RUNTESTS = ctests/test
endif

ifeq ($(MANDIR),$(null))
  MANDIR = $(INSTALLDIR)
endif

ifeq ($(HAVE_SLASHPROC),yes)
  CUFLAGS += -DHAVE_SLASHPROC
endif

ifeq ($(OPENMP),yes)
  CUFLAGS += -DTHREADED_OMP $(COMPFLAG)
else
  ifeq ($(PTHREADS),yes)
    CUFLAGS += -DTHREADED_PTHREADS
  endif
endif

FOBJS =
ifeq ($(FORTRAN),yes)
  FOBJS      = process_namelist.o gptlf.o
  OBJS      += f_wrappers.o
  ifneq ($(ENABLE_CUDA),yes)
    MAKETESTS += ftests/all
    RUNTESTS  += ftests/test
  endif
endif

CUFLAGS += $(INLINEFLAG) $(UNDERSCORING)
ifeq ($(ENABLE_NESTEDOMP),yes)
  CUFLAGS += -DENABLE_NESTEDOMP
endif

ifeq ($(HAVE_MPI),yes)
  CUFLAGS       += -DHAVE_MPI
  FFLAGS       += $(DEFINE)HAVE_MPI
  ifeq ($(HAVE_COMM_F2C),yes)
    CUFLAGS     += -DHAVE_COMM_F2C
  endif
  CUFLAGS       += $(MPI_INCFLAGS)
  LDFLAGS      += $(MPI_LIBFLAGS)
endif

ifeq ($(HAVE_LIBRT),yes)
  CUFLAGS  += -DHAVE_LIBRT
  LDFLAGS += -lrt
endif

ifeq ($(HAVE_NANOTIME),yes)
  CUFLAGS += -DHAVE_NANOTIME
  ifeq ($(BIT64),yes)
    CUFLAGS += -DBIT64
  endif
endif

ifeq ($(HAVE_VPRINTF),yes)
  CUFLAGS += -DHAVE_VPRINTF
endif

ifeq ($(HAVE_TIMES),yes)
  CUFLAGS += -DHAVE_TIMES
endif

ifeq ($(HAVE_GETTIMEOFDAY),yes)
  CUFLAGS += -DHAVE_GETTIMEOFDAY
endif

ALLARGS = lib$(LIBNAME).a
ifeq ($(ENABLE_CUDA),yes)
  OBJS    += print_gpustats.o
  CUFLAGS  +=
endif

ALLARGS += $(MAKETESTS)

##############################################################################
%.o: %.cu
	$(CC) -c $(CUFLAGS) $<

%.o: %.F90
	$(FC) -c $(FFLAGS) $<

all: $(ALLARGS)

cuda/all:
	$(MAKE) -C cuda

acctests/all: cuda/all
	$(MAKE) -C acctests

libonly: lib$(LIBNAME).a 
test: $(RUNTESTS)

# MAKETESTS is ctests/all and maybe ftests/all
ctests/all:
	$(MAKE) -C ctests all

ftests/all:
	$(MAKE) -C ftests all

# RUNTESTS is ctests and maybe ftests
ctests/test:
	$(MAKE) -C ctests test

ftests/test:
	$(MAKE) -C ftests test

lib$(LIBNAME).a: $(OBJS) $(FOBJS)
	$(AR) ruv $@ $(OBJS) $(FOBJS)
	$(RM) -f ctests/*.o ftests/*.o

install: lib$(LIBNAME).a
	install -d $(INSTALLDIR)/lib
	install -d $(INSTALLDIR)/include
	install -d $(INSTALLDIR)/bin
	install -d $(INSTALLDIR)/man/man3 
	install -m 0644 lib$(LIBNAME).a $(INSTALLDIR)/lib
	install -m 0644 gptl.h $(INSTALLDIR)/include
ifeq ($(ENABLE_CUDA),yes)
	install -m 0644 cuda/lib$(LIBNAME)_cuda.a $(INSTALLDIR)/lib
	install -m 0644 cuda/gptl_cuda.h $(INSTALLDIR)/include
	install -m 0644 cuda/gptl_acc.mod $(INSTALLDIR)/include
endif
ifeq ($(FORTRAN),yes)
# *.mod will install either gptl.mod or GPTL.mod
	install -m 0644 gptl.inc *.mod $(INSTALLDIR)/include
endif
	install -m 0644 man/man3/*.3 $(MANDIR)/man/man3
	install -m 0755 *pl $(INSTALLDIR)/bin
ifneq ($(ENABLE_CUDA),yes)
	$(MAKE) -C ctests/ install INSTALLDIR=$(INSTALLDIR)
endif

# Some Fortran compilers name modules in upper case, so account for both possibilities
uninstall:
	$(RM) -f $(INSTALLDIR)/lib/lib$(LIBNAME).a
	$(RM) -f $(INSTALLDIR)/include/gptl.h $(INSTALLDIR)/include/gptl.inc $(INSTALLDIR)/include/gptl.mod $(INSTALLDIR)/include/GPTL.mod
	$(RM) -f $(MANDIR)/man/man3/GPTL*.3

clean:
	$(RM) -f $(OBJS) $(FOBJS) lib$(LIBNAME).a *.mod
	$(MAKE) -C cuda clean
	$(MAKE) -C acctests clean
	$(MAKE) -C ctests clean
	$(MAKE) -C ftests clean

f_wrappers.o: gptl.h private.h devicehost.h
gptl.o: gptl.h private.h devicehost.h
util.o: gptl.h private.h devicehost.h
process_namelist.o: process_namelist.F90 gptl.inc
gptlf.o: gptlf.F90
getoverhead.o: private.h devicehost.h
hashstats.o: private.h devicehost.h
memstats.o: private.h devicehost.h
pr_summary.o: private.h devicehost.h
get_memusage.o: 
print_memusage.o: gptl.h
print_rusage.o: private.h devicehost.h
