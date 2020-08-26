# This is the automake file for GPTL cuda tests.

# Find include directory
AM_CPPFLAGS = -I../include

# Link to GPTL cpu and cuda libraries.
AM_LDFLAGS = ${top_builddir}/${CPU_SRCDIR}/libgptl.a ${top_builddir}/cuda/src/libgptl_cuda.a

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = basic

# Test programs that will be built for all configurations.
check_PROGRAMS = basic
TESTS          = basic

# Hardwire FFLAGS for now
FCFLAGS = -acc -Minfo=accel -ta=tesla:cc60 -m64
