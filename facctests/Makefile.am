# This is the automake file for GPTL cuda tests.

# Hardwire FFLAGS for now
AM_FCFLAGS = $(FCFLAGS) -I../fortran/src -I../cuda/src @FACCFLAGS@

# Link to GPTL cpu and cuda libraries.
# -lstdc++ required for gxx_personality compllaints
AM_LDFLAGS = -L../fortran/src -lgptlf -L../src/.libs -lgptl -L../cuda/src -lgptl_cuda @FACCLDFLAGS@

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = basic badinput verify_results

# Test programs that will be built for all configurations.
check_PROGRAMS = basic badinput
TESTS          = basic badinput

.F90.o:
	$(FC) -c $< $(AM_FCFLAGS)
# Need this rule to enable parallel make
# Why require .mod and cannot use .$(FC_MODEXT) ???
.F90.mod:
	$(FC) -c $< $(AM_FCFLAGS)

# Cannot remember if/why need both getval.o and getval.$(FC_MODEXT) here???
basic.o: basic.F90 ../fortran/src/gptl.$(FC_MODEXT) ../cuda/src/gptl_acc.$(FC_MODEXT)
basic$(EXEEXT): basic.o
	$(FC) -o $@ basic.o $(AM_LDFLAGS)

badinput.o: badinput.F90 ../fortran/src/gptl.$(FC_MODEXT) ../cuda/src/gptl_acc.$(FC_MODEXT)
badinput$(EXEEXT): badinput.o
	$(FC) -o $@ badinput.o $(AM_LDFLAGS)

verify_results.o: verify_results.F90 ../fortran/src/gptl.$(FC_MODEXT) ../cuda/src/gptl_acc.$(FC_MODEXT)
verify_results$(EXEEXT): verify_results.o
	$(FC) -o $@ verify_results.o $(AM_LDFLAGS)
