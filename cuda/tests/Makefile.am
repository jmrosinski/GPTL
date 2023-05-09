# This is the automake file for GPTL cuda tests.

AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/cuda/include

# Link to GPTL cpu and cuda libraries.
AM_LDFLAGS = -L../../src/.libs -lgptl -L../src -lgptl_cuda

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be built but not installed.
noinst_PROGRAMS = basic test1 driver_serial shared verify_init_final

# Test programs that will be built for all configurations.
check_PROGRAMS = basic verify_init_final
TESTS          = basic verify_init_final

basic_SOURCES = basic.cu
basic_OBJS    = basic.o

verify_init_final_SOURCES = verify_init_final.cu
verify_init_final_OBJS    = verify_init_final.o

test1_SOURCES = test1.cu
test1_OBJS    = test1.o

shared_SOURCES = shared.cu
shared_OBJS    = shared.o

driver_serial_SOURCES = driver_serial.cu getval.cu persist.cu sleep1.cu sleep.cu subs.cu
driver_serial_OBJS    = driver_serial.o  getval.o  persist.o  sleep1.o  sleep.o  subs.o 

.cu.o: $(AM_LDFLAGS)
	$(NVCC) -c $< $(AM_CPPFLAGS) $(CUFLAGS)

$(basic_OBJS): $(basic_SOURCES) ../../include/gptl.h ../include/gptl_cuda.h
$(verify_init_final_OBJS): $(verify_init_final_SOURCES) ../../include/gptl.h ../include/gptl_cuda.h
$(test1_OBJS): $(test1_SOURCES) ../../include/gptl.h ../include/gptl_cuda.h
$(driver_serial_OBJS): $(driver_serial_SOURCES) ../../include/gptl.h ../include/gptl_cuda.h localproto.h
$(shared_OBJS): $(shared_SOURCES) ../../include/gptl.h ../include/gptl_cuda.h

basic$(EXEEXT): $(basic_OBJS) ../../include/gptl.h ../include/gptl_cuda.h
	$(NVCC) -o $@ $(basic_OBJS) $(CUFLAGS) $(AM_LDFLAGS)
verify_init_final$(EXEEXT): $(verify_init_final_OBJS) ../../include/gptl.h ../include/gptl_cuda.h
	$(NVCC) -o $@ $(verify_init_final_OBJS) $(CUFLAGS) $(AM_LDFLAGS)
test1$(EXEEXT): $(test1_OBJS) ../../include/gptl.h ../include/gptl_cuda.h
	$(NVCC) -o $@ $(test1_OBJS) $(CUFLAGS) $(AM_LDFLAGS)
driver_serial$(EXEEXT):  $(driver_serial_OBJS)
	$(NVCC) -o $@ $(driver_serial_OBJS) $(CUFLAGS) $(AM_LDFLAGS)
shared$(EXEEXT):  $(shared_OBJS)
	$(NVCC) -o $@ $(shared_OBJS) $(CUFLAGS) $(AM_LDFLAGS)

CLEANFILES = *.ptx *.ii *.module_id *.cudafe1.* *.fatbin*
