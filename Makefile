include macros.make

ifeq ($(findstring xlf, $(FC)),xlf)
  DEFINE = -WF,-D
else
  DEFINE = -D
endif

null =
OBJS = gptl.o util.o threadutil.o get_memusage.o print_memusage.o \
       gptl_papi.o pmpi.o

ifeq ($(ENABLE_PMPI),yes)
  CFLAGS += -DENABLE_PMPI
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
  CFLAGS  += -DTHREADED_OMP $(COMPFLAG)
else
  ifeq ($(PTHREADS),yes)
    CFLAGS  += -DTHREADED_PTHREADS
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
  CFLAGS_TESTS += -DHAVE_MPI
  CFLAGS       += $(MPI_INCFLAGS)
  CFLAGS_TESTS += $(MPI_INCFLAGS)
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

##############################################################################

all: lib$(LIBNAME).a $(MAKETESTS)
libonly: lib$(LIBNAME).a 
test: $(RUNTESTS)

# MAKETESTS is ctests/all and maybe ftests/all
ctests/all:
	(cd ctests && $(MAKE) all)

ftests/all:
	(cd ftests && $(MAKE) all)

# RUNTESTS is ctests and maybe ftests
ctests/test:
	(cd ctests && $(MAKE) test)

ftests/test:
	(cd ftests && $(MAKE) test)

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
	install -m 0644 gptl.inc gptl.mod $(INSTALLDIR)/include
endif
	install -m 0644 man/man3/*.3 $(MANDIR)/man/man3
	install -m 0755 *pl $(INSTALLDIR)/bin
	(cd ctests/ && make install INSTALLDIR=$(INSTALLDIR))

# Some Fortran compilers name modules in upper case, so account for both possibilities
uninstall:
	$(RM) -f $(INSTALLDIR)/lib/lib$(LIBNAME).a
	$(RM) -f $(INSTALLDIR)/include/gptl.h $(INSTALLDIR)/include/gptl.inc $(INSTALLDIR)/include/gptl.mod $(INSTALLDIR)/include/GPTL.mod
	$(RM) -f $(MANDIR)/man/man3/GPTL*.3

clean:
	$(RM) -f $(OBJS) $(FOBJS) lib$(LIBNAME).a *.mod
	(cd ctests && $(MAKE) clean)
	(cd ftests && $(MAKE) clean)

f_wrappers.o: gptl.h private.h
f_wrappers_pmpi.o: gptl.h private.h
gptl.o: gptl.h private.h
util.o: gptl.h private.h
threadutil.o: gptl.h private.h
gptl_papi.o: gptl.h private.h
process_namelist.o: process_namelist.F90 gptl.inc
	$(FC) -c $(FFLAGS) $<
gptlf.o: gptlf.F90
	$(FC) -c $(FFLAGS) $<
pmpi.o: gptl.h private.h
