# This is the automake file for GPTL cuda tests.

CC = nvc

# -acc gpu says to recognize OpenACC directives
AM_CFLAGS = -I${top_srcdir} -I../include -I../cuda/include -acc gpu

# Link to GPTL cpu and cuda libraries.
# -cudalib says link in all necessary cuda libraries
AM_LDFLAGS = -cudalib -L${top_builddir}/${CPU_SRCDIR} -lgptl -L${top_builddir}/cuda/src -lgptl_cuda \
             -ta=tesla,cc60

# Need unwind library if configure was set up that way
if HAVE_LIBUNWIND
  AM_LDFLAGS += -lunwind
endif

# These programs will be build but not installed
noinst_PROGRAMS = basic

# Test programs that will be built for all configurations.
check_PROGRAMS = basic
TESTS          = basic

basic_SOURCES = basic.cu
basic_OBJS    = basic.o

.c.o: $(AM_LDFLAGS)
	$(CC) -c $< $(AM_CFLAGS)

basic$(EXEEXT): $(basic_OBJS)
	$(CC) -o $@ $(basic_OBJS) $(AM_LDFLAGS)

$(basic_OBJS): $(basic_SOURCES)
