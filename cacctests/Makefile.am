# This is the automake file for GPTL cuda tests.

# -acc gpu says to recognize OpenACC directives
AM_CFLAGS = -I${top_srcdir} -I../include -I../cuda/include @CACCFLAGS@

# Link to GPTL cpu and cuda libraries.
# -cudalib says link in all necessary cuda libraries
AM_LDFLAGS = -L../src/.libs -lgptl -L../cuda/src -lgptl_cuda @CACCLDFLAGS@

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be build but not installed
noinst_PROGRAMS = basic ohdest warp2sm

# Test programs that will be built for all configurations.
check_PROGRAMS = basic
TESTS          = basic

basic_SOURCES = basic.c
basic_OBJS    = basic.o

ohdest_SOURCES = ohdest.c
ohdest_OBJS    = ohdest.o

warp2sm_SOURCES = warp2sm.c
warp2sm_OBJS    = warp2sm.o

.c.o: $(AM_LDFLAGS)
	$(CC) -c $< $(AM_CFLAGS)

basic$(EXEEXT): $(basic_OBJS)
	$(CC) -o $@ $(basic_OBJS) $(AM_LDFLAGS)

ohdest$(EXEEXT): $(ohdest_OBJS)
	$(CC) -o $@ $(ohdest_OBJS) $(AM_LDFLAGS)

warp2sm$(EXEEXT): $(warp2sm_OBJS)
	$(CC) -o $@ $(warp2sm_OBJS) $(AM_LDFLAGS)

$(basic_OBJS): $(basic_SOURCES) ../include/gptl.h ../cuda/include/gptl_acc.h
$(warp2sm_OBJS): $(warp2sm_SOURCES) ../include/gptl.h ../cuda/include/gptl_acc.h
