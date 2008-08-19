# Initialize some macros to null before including arch-specific settings.
# CFLAGS_TESTS differs from CFLAGS for optimization (esp. inlining), and leaving off
# unused library-specific settings.

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
TESTS = ctests/all 

LDFLAGS += $(ABIFLAGS)

ifeq ($(MANDIR),$(null))
  MANDIR = $(INSTALLDIR)
endif

ifeq ($(LINUX),yes)
  CFLAGS += -DLINUX
endif

ifeq ($(OPENMP),yes)
  CFLAGS  += -DTHREADED_OMP $(COMPFLAG)
  CFLAGS_TESTS += -DTHREADED_OMP $(COMPFLAG)
  LDFLAGS += $(COMPFLAG)
  FFLAGS  += $(FOMPFLAG)
else
  ifeq ($(PTHREADS),yes)
    CFLAGS  += -DTHREADED_PTHREADS
    LDFLAGS += -lpthread
  endif
endif

ifeq ($(FORTRAN),yes)
 OBJS  += f_wrappers.o
 TESTS += ftests/all
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
  CFLAGS_TESTS += -DHAVE_MPI
  CFLAGS       += $(MPI_INCFLAGS)
  CFLAGS_TESTS += $(MPI_INCFLAGS)
  LDFLAGS      += $(MPI_LIBFLAGS)
else
  HAVE_MPI = no
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
	install -m 0644 man/man3/*.3 $(MANDIR)/man/man3
	install -m 0755 *pl $(INSTALLDIR)/bin

uninstall:
	rm -f $(INSTALLDIR)/lib/lib$(LIBNAME).a
	rm -f $(INSTALLDIR)/include gptl.h $(INSTALLDIR)/include/gptl.inc
	rm -f $(MANDIR)/man/man3/GPTL*.3

ctests/all:
	(cd ctests && $(MAKE) all CC=$(CC) HAVE_MPI=$(HAVE_MPI) HAVE_PAPI=$(HAVE_PAPI) \
        CFLAGS="$(CFLAGS_TESTS)" LDFLAGS="$(LDFLAGS)")

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
