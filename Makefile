include macros.make

null =
OBJS = gptl.o util.o threadutil.o get_memusage.o print_memusage.o gptl_papi.o

ifeq ($(DEBUG),yes)
  LIBNAME = gptl_debug
else
  LIBNAME = gptl
endif

LDFLAGS = -L.. -l$(LIBNAME)
TESTS = ctests/all 

# Set variables based on settings in macros.make

LDFLAGS += $(ABIFLAGS)

ifeq ($(LINUX),yes)
  CFLAGS       += -DLINUX
endif

ifeq ($(FORTRAN),yes)
 OBJS += f_wrappers.o
 TESTS += ftests/all
endif

ifeq ($(OPENMP),yes)
  CFLAGS  += -DTHREADED_OMP $(COMPFLAG)
  LDFLAGS += $(COMPFLAG)
  FFLAGS  += $(FOMPFLAG)
endif

CFLAGS += $(INLINEFLAG) $(UNDERSCORING)

ifeq ($(HAVE_PAPI),yes)
  CFLAGS       += -DHAVE_PAPI
  CFLAGS_TESTS += -DHAVE_PAPI
  ifneq ($(PAPI_INCFLAGS),$(null))
    CFLAGS       += $(PAPI_INCFLAGS)
    CFLAGS_TESTS += $(PAPI_INCFLAGS)
  endif
  ifneq ($(PAPI_LIBFLAGS),$(null))
    LDFLAGS += $(PAPI_LIBFLAGS)
  endif
endif

ifeq ($(HAVE_MPI),yes)
  CFLAGS       += -DHAVE_MPI
  CFLAGS_TESTS += -DHAVE_MPI
  ifneq ($(MPI_INCFLAGS),$(null))
    CFLAGS       += $(MPI_INCFLAGS)
    CFLAGS_TESTS += $(MPI_INCFLAGS)
  endif
  ifneq ($(MPI_LIBFLAGS),$(null))
    LDFLAGS += $(MPI_LIBFLAGS)
  endif
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

all: lib$(LIBNAME).a $(TESTS)
libonly: lib$(LIBNAME).a 

lib$(LIBNAME).a: $(OBJS)
	$(AR)  ruv $@ $(OBJS)
	rm -f ctests/*.o ftests/*.o

install: lib$(LIBNAME).a
	install -m 0644 lib$(LIBNAME).a $(INSTALLDIR)/lib
	install -m 0644 gptl.h gptl.inc $(INSTALLDIR)/include
	install -m 0644 man/man3/*.3 $(INSTALLDIR)/man/man3

uninstall:
	rm -f $(INSTALLDIR)/lib/lib$(LIBNAME).a
	rm -f $(INSTALLDIR)/include gptl.h $(INSTALLDIR)/include/gptl.inc
	rm -f $(MANDIR)/GPTL*

ctests/all:
	(cd ctests && $(MAKE) all CC=$(CC) CFLAGS="$(CFLAGS_TESTS)" LDFLAGS="$(LDFLAGS)")

ftests/all:
	(cd ftests && $(MAKE) all FC=$(FC) FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)")

clean:
	rm -f $(OBJS) lib$(LIBNAME).a
	(cd ctests && $(MAKE) clean)
	(cd ftests && $(MAKE) clean)

f_wrappers.o: f_wrappers.c gptl.h private.h
gptl.o: gptl.c gptl.h private.h
util.o: util.c gptl.h private.h
threadutil.o: threadutil.c gptl.h private.h
gptl_papi.o: gptl_papi.c gptl.h private.h
