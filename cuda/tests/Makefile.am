# This is the automake file for GPTL cuda tests.

# Find include directory
AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/cuda/include

# Link to GPTL cpu and cuda libraries.
AM_LDFLAGS = ${top_builddir}/src_cuda/libgptl.a ${top_builddir}/cuda/src/libgptl_cuda.a

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = test1 driver_serial

# Test programs that will be built for all configurations.
check_PROGRAMS = test1 driver_serial
TESTS          = test1 driver_serial

test1_SOURCES = test1.cu
test1_OBJS    = test1.o
driver_serial_SOURCES = driver_serial.cu getval.cu persist.cu sleep1.cu sleep.cu subs.cu
driver_serial_OBJS    = driver_serial.o  getval.o  persist.o  sleep1.o  sleep.o  subs.o 

.cu.o: $(AM_LDFLAGS)
	$(CC) -c $< $(AM_CPPFLAGS) $(CUFLAGS)

test1$(EXEEXT): $(test1_OBJS)
	$(CC) -o $@ $(test1_OBJS) $(CUFLAGS) $(AM_LDFLAGS)
driver_serial$(EXEEXT):  $(driver_serial_OBJS)
	$(CC) -o $@ $(driver_serial_OBJS) $(CUFLAGS) $(AM_LDFLAGS)
