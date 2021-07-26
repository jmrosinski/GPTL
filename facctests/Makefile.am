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
noinst_PROGRAMS = basic driver_vdmints3 warp2sm utrtest

# Test programs that will be built for all configurations.
check_PROGRAMS = basic
TESTS          = basic

driver_vdmints3_SOURCES = driver_vdmints3.cu vdmints3_sim.cu getval.cu
driver_vdmints3_OBJS =    driver_vdmints3.o  vdmints3_sim.o  getval.o

warp2sm_SOURCES = warp2sm.F90
warp2sm_OBJS    = warp2sm.o

utrtest_SOURCES = utrtest.F90
utrtest_OBJS    = utrtest.o

.F90.o:
	$(FC) -c $< $(AM_FCFLAGS)
# Need this rule to enable parallel make
# Why require .mod and cannot use .$(FC_MODEXT) ???
.F90.mod:
	$(FC) -c $< $(AM_FCFLAGS)

# Cannot remember if/why need both getval.o and getval.$(FC_MODEXT) here???
basic.o: basic.F90 getval.o getval.$(FC_MODEXT) ../fortran/src/gptl.$(FC_MODEXT) ../cuda/src/gptl_acc.mod
basic$(EXEEXT): basic.o
	$(FC) -o $@ basic.o $(AM_LDFLAGS)

driver_vdmints3.o: getval.$(FC_MODEXT) ../fortran/src/gptl.$(FC_MODEXT) ../cuda/src/gptl_acc.mod
driver_vdmints3$(EXEEXT): $(driver_vdmints3_OBJS)
	$(FC) -o $@ $(driver_vdmints3_OBJS) $(AM_LDFLAGS)

warp2sm.o: warp2sm.F90
warp2sm$(EXEEXT): warp2sm.o
	$(FC) -o $@ warp2sm.o $(AM_LDFLAGS)

utrtest.o: utrtest.F90
utrtest$(EXEEXT): utrtest.o
	$(FC) -o $@ utrtest.o $(AM_LDFLAGS)
