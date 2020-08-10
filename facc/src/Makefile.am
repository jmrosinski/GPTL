# This automake file generates the Makefile to build the GPTL cuda library.

# This is our output. The GPTL library of Fortran wrapping routines for cuda
lib_LTLIBRARIES = libgptlf_cuda.la

# Follow libtool versioning rules (c:r:a) (current:revision:age)
# lib source changes since last update => increment "r"
# lib API added, removed, or changed since last update => increment "c" and set "r"=0
# API added since last public release => increment "a"
# API removed or changed since last public release => set "a"=0
libgptlf_cuda_la_LDFLAGS = -version-info 0:0:0

# Maximum register count
MAXRREGCOUNT = 8

# These are the source files.
libgptlf_cuda_la_SOURCES = gptl_acc.F90

# Using -Mcuda=cc60 instead of -Mcuda=cuda8 fixes the static memory address compiler bug
FFLAGS  = -g -O0 -I../include -acc -Minfo=accel -Minfo -ta=tesla:cc60,keepgpu -m64
