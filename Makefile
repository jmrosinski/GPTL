# CFLAGS_TESTS differs from CFLAGS for optimization (esp. inlining), and 
# leaving off unused library-specific settings.

CFLAGS_TESTS = -g

include macros.make

null =
OBJS = gptl.o util.o threadutil.o get_memusage.o print_memusage.o gptl_papi.o

ifeq ($(DEBUG),yes)
  LIBNAME = gptl_debug
else
  LIBNAME = gptl
endif

LDFLAGS = -L.. -l$(LIBNAME)
MAKETESTS = ctests/all
RUNTESTS = ctests/test

LDFLAGS += $(ABIFLAGS)

ifeq ($(MPICMD),$(null))
  MPICMD = mpiexec
endif

ifeq ($(MANDIR),$(null))
  MANDIR = $(INSTALLDIR)
endif

ifeq ($(LINUX),yes)
  CFLAGS += -DLINUX
endif

ifeq ($(OPENMP),yes)
  CFLAGS  += -DTHREADED_OMP $(COMPFLAG)
  CFLAGS_TESTS += -DTHREADED_OMP
else
  ifeq ($(PTHREADS),yes)
    CFLAGS  += -DTHREADED_PTHREADS
    LDFLAGS += -lpthread
  endif
endif
FFLAGS += $(FOMPFLAG)
CFLAGS_TESTS += $(COMPFLAG)
# Assume C and Fortran OMP flags are the same
LDFLAGS += $(COMPFLAG)

FOBJS =
ifeq ($(FORTRAN),yes)
  FOBJS      = gptlprocess_namelist.o
  OBJS      += f_wrappers.o
  MAKETESTS += ftests/all
  RUNTESTS  += ftests/test
endif

CFLAGS += $(INLINEFLAG) $(UNDERSCORING)

ifeq ($(HAVE_PAPI),yes)
  CFLAGS       += -DHAVE_PAPI
  CFLAGS_TESTS += -DHAVE_PAPI
  CFLAGS       += $(PAPI_INCFLAGS)
  CFLAGS_TESTS += $(PAPI_INCFLAGS)
  LDFLAGS      += $(PAPI_LIBFLAGS)
else
  HAVE_PAPI = no
endif

ifeq ($(HAVE_MPI),yes)
  CFLAGS       += -DHAVE_MPI
  ifeq ($(HAVE_COMM_F2C),yes)
    CFLAGS     += -DHAVE_COMM_F2C
  endif
  CFLAGS_TESTS += -DHAVE_MPI
  CFLAGS       += $(MPI_INCFLAGS)
  CFLAGS_TESTS += $(MPI_INCFLAGS)
  LDFLAGS      += $(MPI_LIBFLAGS)
  HEADER        = gptl.h.havempi
else
  HAVE_MPI = no
  HEADER   = gptl.h.nompi
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

ifeq ($(INSTRFLAG),$(null))
  INSTRFLAG = -zzz
endif

##############################################################################

all: lib$(LIBNAME).a $(MAKETESTS)
libonly: lib$(LIBNAME).a 
test: $(RUNTESTS)

# MAKETESTS is ctests/all and maybe ftests/all
ctests/all:
	(cd ctests && $(MAKE) all CC=$(CC) CXX=$(CXX) HAVE_MPI=$(HAVE_MPI) \
         HAVE_PAPI=$(HAVE_PAPI) CFLAGS="$(CFLAGS_TESTS)" LDFLAGS="$(LDFLAGS)" \
         TEST_AUTOPROFILE=$(TEST_AUTOPROFILE) INSTRFLAG=$(INSTRFLAG))

ftests/all:
	(cd ftests && $(MAKE) all FC=$(FC) FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)" \
         HAVE_MPI=$(HAVE_MPI) HAVE_PAPI=$(HAVE_PAPI))

# RUNTESTS is ctests and maybe ftests
ctests/test:
	(cd ctests && $(MAKE) test MPICMD=$(MPICMD) HAVE_MPI=$(HAVE_MPI) \
         HAVE_PAPI=$(HAVE_PAPI) TEST_AUTOPROFILE=$(TEST_AUTOPROFILE))

ftests/test:
	(cd ftests && $(MAKE) test MPICMD=$(MPICMD) HAVE_MPI=$(HAVE_MPI) \
         HAVE_PAPI=$(HAVE_PAPI))

lib$(LIBNAME).a: $(OBJS) $(FOBJS) gptl.h
	$(AR) ruv $@ $(OBJS) $(FOBJS)
	$(RM) -f ctests/*.o ftests/*.o

install: lib$(LIBNAME).a
	install -m 0644 lib$(LIBNAME).a $(INSTALLDIR)/lib
	install -m 0644 gptl.h gptl.inc $(INSTALLDIR)/include
	install -m 0644 man/man3/*.3 $(MANDIR)/man/man3
	install -m 0755 *pl $(INSTALLDIR)/bin
	(cd ctests/ && make install INSTALLDIR=$(INSTALLDIR))

uninstall:
	$(RM) -f $(INSTALLDIR)/lib/lib$(LIBNAME).a
	$(RM) -f $(INSTALLDIR)/include/gptl.h $(INSTALLDIR)/include/gptl.inc
	$(RM) -f $(MANDIR)/man/man3/GPTL*.3

clean:
	$(RM) -f $(OBJS) $(FOBJS) lib$(LIBNAME).a gptl.h
	(cd ctests && $(MAKE) clean HAVE_MPI=$(HAVE_MPI) HAVE_PAPI=$(HAVE_PAPI) \
	 TEST_AUTOPROFILE=$(TEST_AUTOPROFILE))
	(cd ftests && $(MAKE) clean HAVE_MPI=$(HAVE_MPI) HAVE_PAPI=$(HAVE_PAPI))

gptl.h: $(HEADER)
	cp -f $(HEADER) gptl.h
f_wrappers.o: f_wrappers.c gptl.h private.h
gptl.o: gptl.c gptl.h private.h
util.o: util.c gptl.h private.h
threadutil.o: threadutil.c gptl.h private.h
gptl_papi.o: gptl_papi.c gptl.h private.h
gptlprocess_namelist.o: gptlprocess_namelist.F90 gptl.inc
	$(FC) -c $(FFLAGS) $<
