# This is the automake file for GPTL cuda tests.

# For now hardwire the compiler
NVCC = nvcc

# Find include directory
AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/cuda/include

# Link to GPTL cpu and cuda libraries.
AM_LDFLAGS = ${top_builddir}/src/.libs/libgptl.a ${top_builddir}/cuda/src/libgptl_cuda.a

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = test1

# Test programs that will be built for all configurations.
check_PROGRAMS = test1
TESTS          = test1

test1_SOURCES = test1.cu
#driver_serial_SOURCES = getval.cu driver_serial.cu persist.cu sleep1.cu sleep.cu subs.cu
test1_LDADD = ../../src/libgptl.la ../src/libgptl_cuda.a
#driver_serial_LDADD = ../../src/.libs/libgptl.la ../src/libgptl_cuda.a

# For now just hardwire the compile flags
CUFLAGS = -g -ftz=true -gencode arch=compute_60,code=compute_60 -rdc=true --maxrregcount 32

.cu.o: ../src/libgptl_cuda.a
	$(NVCC) -c $< $(AM_CPPFLAGS) $(CUFLAGS)

test1$(EXEEXT): test1.o $(test1_LDADD)
	$(CC) -o $@ test1.o $(CFLAGS) $(AM_LDFLAGS)
