# This is part of Unidata's GPTL package.  Copyright 2018, see
# the COPYRIGHT file for more information.

# This automake file generates the Makefile to build the GPTL C library.
# Ed Hartnett 5/18/18

AM_CPPFLAGS = -I$(top_srcdir)/include 

libgptl_la_CPPFLAGS = ${AM_CPPFLAGS}

# This is our output. The GPTL library.
lib_LTLIBRARIES = libgptl.la
libgptl_la_SOURCES = f_wrappers.c getoverhead.c gptl.c gptl_papi.c	\
hashstats.c memstats.c memusage.c pmpi.c print_rusage.c pr_summary.c	\
util.c

