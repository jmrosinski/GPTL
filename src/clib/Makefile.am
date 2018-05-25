# This is part of NOAAs GPTL package.

# This automake file generates the Makefile to build the GPTL C library.
# Ed Hartnett 5/18/18

libgptl_la_CPPFLAGS = -I$(top_srcdir)/include 

# This is our output. The GPTL library.
lib_LTLIBRARIES = libgptl.la

# These are the source files.
libgptl_la_SOURCES = f_wrappers.c getoverhead.c gptl.c gptl_papi.c	\
hashstats.c memstats.c memusage.c pmpi.c print_rusage.c pr_summary.c	\
util.c

