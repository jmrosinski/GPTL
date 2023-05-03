# This is the automake file for GPTL cuda tests.

AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/cuda/include

# Link to GPTL cpu and cuda libraries.
AM_LDFLAGS = -L../../src/.libs -lgptl -L../src -lgptl_cuda

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = basic

# Test programs that will be built for all configurations.
check_PROGRAMS = basic
TESTS          = basic

basic_SOURCES = basic.cu
basic_OBJS    = basic.o

.cu.o: $(AM_LDFLAGS)
	$(NVCC) -c $< $(AM_CPPFLAGS) $(CUFLAGS)

$(basic_OBJS): $(basic_SOURCES) ../../include/gptl.h ../include/gptl_cuda.h
basic$(EXEEXT): $(basic_OBJS) ../../include/gptl.h ../include/gptl_cuda.h
	$(NVCC) -o $@ $(basic_OBJS) $(CUFLAGS) $(AM_LDFLAGS)

CLEANFILES = *.ptx *.ii *.module_id *.cudafe1.* *.fatbin*
