# This is the automake file for GPTL cuda tests.

AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/cuda/include

# Link to GPTL cpu and cuda libraries.
AM_LDFLAGS = -L../../src/.libs -lgptl -L../src -lgptl_cuda

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = basic test1 driver_serial

# Test programs that will be built for all configurations.
check_PROGRAMS = basic
TESTS          = basic

basic_SOURCES = basic.cu
basic_OBJS    = basic.o

test1_SOURCES = test1.cu
test1_OBJS    = test1.o

driver_serial_SOURCES = driver_serial.cu getval.cu persist.cu sleep1.cu sleep.cu subs.cu
driver_serial_OBJS    = driver_serial.o  getval.o  persist.o  sleep1.o  sleep.o  subs.o 

.cu.o: $(AM_LDFLAGS)
	$(NVCC) -c $< $(AM_CPPFLAGS) $(CUFLAGS)

$(basic_OBJS): $(basic_SOURCES)
$(test1_OBJS): $(test1_SOURCES)
$(driver_serial_OBJS): $(driver_serial_SOURCES)

basic$(EXEEXT): $(basic_OBJS)
	$(NVCC) -o $@ $(basic_OBJS) $(CUFLAGS) $(AM_LDFLAGS)
test1$(EXEEXT): $(test1_OBJS)
	$(NVCC) -o $@ $(test1_OBJS) $(CUFLAGS) $(AM_LDFLAGS)
driver_serial$(EXEEXT):  $(driver_serial_OBJS)
	$(NVCC) -o $@ $(driver_serial_OBJS) $(CUFLAGS) $(AM_LDFLAGS)
