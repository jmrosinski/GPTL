# This is the automake file for GPTL cuda tests.

# Hardwire FFLAGS for now
AM_FCFLAGS = $(FCFLAGS) -I../fortran/src -I../cuda/src -acc -Minfo=accel -ta=tesla:cc60 -m64

# Link to GPTL cpu and cuda libraries.
# -Mcuda=cc60 required for CUDA functions
# -lstdc++ required for gxx_personality compllaints
AM_LDFLAGS = ../fortran/src/libgptlf.a ../src_nvcc/libgptl.a ../cuda/src/libgptl_cuda.a \
             -Mcuda=cc60 -lstdc++

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = basic

# Test programs that will be built for all configurations.
check_PROGRAMS = basic
TESTS          = basic

.F90.o:
	$(FC) -c $< $(AM_FCFLAGS)

basic.o: basic.F90

basic$(EXEEXT): basic.o
	$(FC) -o $@ basic.o $(AM_FCFLAGS) $(AM_LDFLAGS)
