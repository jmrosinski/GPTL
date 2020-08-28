# This is the automake file for GPTL cuda tests.

AM_CPPFLAGS = -I${top_srcdir} -I../include -I../cuda/include

# Link to GPTL cpu and cuda libraries.
AM_LDFLAGS = ${top_builddir}/${CPU_SRCDIR}/libgptl.a ${top_builddir}/cuda/src/libgptl_cuda.a

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be build but not installed
noinst_PROGRAMS = basic

# Test programs that will be built for all configurations.
check_PROGRAMS = basic
TESTS          = basic

basic_SOURCES = basic.cu
basic_OBJS    = basic.o

.cu.o: $(AM_LDFLAGS)
	$(CC) -c $< $(AM_CPPFLAGS) $(CUFLAGS)

basic$(EXEEXT): $(basic_OBJS)
	$(CC) -o $@ $(basic_OBJS) $(CUFLAGS) $(AM_LDFLAGS)

$(basic_OBJS): $(basic_SOURCES)
