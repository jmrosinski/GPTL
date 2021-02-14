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
noinst_PROGRAMS = basic

# Test programs that will be built for all configurations.
check_PROGRAMS = basic
TESTS          = basic

basic_SOURCES = basic.c
basic_OBJS    = basic.o

.c.o: $(AM_LDFLAGS)
	$(CC) -c $< $(AM_CFLAGS)

basic$(EXEEXT): $(basic_OBJS)
	$(CC) -o $@ $(basic_OBJS) $(AM_LDFLAGS)

$(basic_OBJS): $(basic_SOURCES) ../include/gptl.h ../cuda/include/gptl_acc.h
