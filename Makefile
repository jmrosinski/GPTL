include macros.make

ifeq ($(findstring xlf, $(FC)),xlf)
  DEFINE = -WF,-D
else
  DEFINE = -D
endif

null =
OBJS = gptl.o util.o memusage.o gptl_papi.o pmpi.o getoverhead.o \
       hashstats.o memstats.o pr_summary.o print_rusage.o

ifeq ($(ENABLE_PMPI),yes)
  CFLAGS += -DENABLE_PMPI -DMPI_STATUS_SIZE_IN_INTS=$(MPI_STATUS_SIZE_IN_INTS)
  ifeq ($(MPI_CONST),yes)
    CFLAGS += -DCONST=const
  else
    CFLAGS += -DCONST=
  endif
  ifeq ($(HAVE_IARGCGETARG),yes)
    CFLAGS += -DHAVE_IARGCGETARG
  endif
  LIBNAME = gptl_pmpi
else
  LIBNAME = gptl
endif

# Always run the C tests. Add Fortran tests if Fortran enabled
MAKETESTS = ctests/all
RUNTESTS = ctests/test

ifeq ($(MANDIR),$(null))
  MANDIR = $(INSTALLDIR)
endif

ifeq ($(HAVE_SLASHPROC),yes)
  CFLAGS += -DHAVE_SLASHPROC
endif

ifeq ($(OPENMP),yes)
  CFLAGS += -DTHREADED_OMP $(COMPFLAG)
else
  ifeq ($(PTHREADS),yes)
    CFLAGS += -DTHREADED_PTHREADS
  endif
endif

FOBJS =
ifeq ($(FORTRAN),yes)
  FOBJS      = process_namelist.o gptlf.o
  OBJS      += f_wrappers.o f_wrappers_pmpi.o
  MAKETESTS += ftests/all
  RUNTESTS  += ftests/test
endif

CFLAGS += $(INLINEFLAG) $(UNDERSCORING)
ifeq ($(ENABLE_NESTEDOMP),yes)
  CFLAGS += -DENABLE_NESTEDOMP
endif

ifeq ($(HAVE_PAPI),yes)
  CFLAGS += -DHAVE_PAPI
  CFLAGS += $(PAPI_INCFLAGS)
  FFLAGS += $(DEFINE)HAVE_PAPI
endif

ifeq ($(HAVE_MPI),yes)
  CFLAGS       += -DHAVE_MPI
  FFLAGS       += $(DEFINE)HAVE_MPI
  ifeq ($(HAVE_COMM_F2C),yes)
    CFLAGS     += -DHAVE_COMM_F2C
  endif
  CFLAGS       += $(MPI_INCFLAGS)
  LDFLAGS      += $(MPI_LIBFLAGS)
endif

ifeq ($(HAVE_LIBRT),yes)
  CFLAGS  += -DHAVE_LIBRT
  LDFLAGS += -lrt
endif

ifeq ($(HAVE_NANOTIME),yes)
  CFLAGS += -DHAVE_NANOTIME
  ifeq ($(BIT64),yes)
    CFLAGS += -DBIT64
  endif
endif

ifeq ($(HAVE_VPRINTF),yes)
  CFLAGS += -DHAVE_VPRINTF
endif

ifeq ($(HAVE_TIMES),yes)
  CFLAGS += -DHAVE_TIMES
endif

ifeq ($(HAVE_GETTIMEOFDAY),yes)
  CFLAGS += -DHAVE_GETTIMEOFDAY
endif

ALLARGS = lib$(LIBNAME).a
ifeq ($(ENABLE_GPU),yes)
  OBJS    += sub1_.o
  CFLAGS  += -acc -Minfo=accel -Minfo -ta=tesla:cc60
  ALLARGS += acctests2/all acctests/all
endif

ALLARGS += $(MAKETESTS)

ifeq ($(FORTRAN),yes)
  ALLARGS += printmpistatussize
endif

##############################################################################
%.o: %.F90
	$(FC) -c $(FFLAGS) $<

all: $(ALLARGS)
ifeq ($(FORTRAN),yes)
printmpistatussize: printmpistatussize.o
	$(FC) -o $@ $? $(FFLAGS)
endif

cuda/all:
	$(MAKE) -C cuda

acctests2/all: cuda/all
	$(MAKE) -C acctests2

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
ifeq ($(FORTRAN),yes)
# *.mod will install either gptl.mod or GPTL.mod
	install -m 0644 gptl.inc *.mod $(INSTALLDIR)/include
endif
	install -m 0644 man/man3/*.3 $(MANDIR)/man/man3
	install -m 0755 *pl $(INSTALLDIR)/bin
	$(MAKE) -C ctests/ install INSTALLDIR=$(INSTALLDIR)

# Some Fortran compilers name modules in upper case, so account for both possibilities
uninstall:
	$(RM) -f $(INSTALLDIR)/lib/lib$(LIBNAME).a
	$(RM) -f $(INSTALLDIR)/include/gptl.h $(INSTALLDIR)/include/gptl.inc $(INSTALLDIR)/include/gptl.mod $(INSTALLDIR)/include/GPTL.mod
	$(RM) -f $(MANDIR)/man/man3/GPTL*.3

clean:
	$(RM) -f $(OBJS) $(FOBJS) lib$(LIBNAME).a *.mod printmpistatussize.o printmpistatussize
	$(MAKE) -C cuda clean
	$(MAKE) -C acctests2 clean
	$(MAKE) -C acctests clean
	$(MAKE) -C ctests clean
	$(MAKE) -C ftests clean

f_wrappers.o: gptl.h private.h devicehost.h
f_wrappers_pmpi.o: gptl.h private.h devicehost.h
gptl.o: gptl.h private.h devicehost.h
util.o: gptl.h private.h devicehost.h
gptl_papi.o: gptl.h private.h devicehost.h
process_namelist.o: process_namelist.F90 gptl.inc
gptlf.o: gptlf.F90
pmpi.o: gptl.h private.h devicehost.h
getoverhead.o: private.h devicehost.h
hashstats.o: private.h devicehost.h
memstats.o: private.h devicehost.h
pr_summary.o: private.h devicehost.h
get_memusage.o: 
print_memusage.o: gptl.h
print_rusage.o: private.h devicehost.h

printmpistatussize.o: printmpistatussize.F90
