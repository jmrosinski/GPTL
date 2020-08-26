# This is the automake file for GPTL cuda tests.

# Find include directory
AM_CPPFLAGS = -I../include

# Link to GPTL cpu and cuda libraries.
AM_LDFLAGS = ${top_builddir}/${CPU_SRCDIR}/libgptl.la ${top_builddir}/cuda/src/libgptlf_cuda.la ${top_builddir}/cuda/src/libgptl_cuda.la

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = driver_serial vdmints3

# Test programs that will be built for all configurations.
check_PROGRAMS = driver_serial vdmints3
TESTS          = driver_serial vdmints3

# Hardwire FFLAGS for now
FFLAGS  = -g -O0 -I../include -acc -Minfo=accel -Minfo -ta=tesla:cc60,keepgpu -m64
