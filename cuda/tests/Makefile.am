# This is the automake file for GPTL cuda tests.

# For now hardwire the compiler
CC = nvcc

# Find include directory
AM_CPPFLAGS = -I../include

# Link to GPTL cpu and cuda libraries.
AM_LDFLAGS = ${top_builddir}/src/libgptl.la ${top_builddir}/cuda/src/libgptl_cuda.la

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = driver_serial test1

# Test programs that will be built for all configurations.
check_PROGRAMS = driver_serial test1
TESTS          = driver_serial test1

test1_SOURCES = test1.cu
driver_serial_SOURCES = getval.cu driver_serial.cu persist.cu sleep1.cu sleep.cu subs.cu

# For now just hardwire the compile flags
CFLAGS = -c -g -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true --maxrregcount 32

.cu.o:
	$(CC) -c $< $(CFLAGS)
