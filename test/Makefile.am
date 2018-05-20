## This is a automake file, part of the GPTL package.
# Copyright 2018, see the COPYRIGHT file for more information.

# This is the automake file for GPTL C tests.
# Ed Hartnett, 5/18/18

# Find include directory.
AM_CPPFLAGS = -I$(top_srcdir)/include

# Link to GPTL library.
AM_LDFLAGS = ${top_builddir}/src/clib/libgptl.la

# Test programs to be built.
check_PROGRAMS = tst_simple

# Tests to be run.
TESTS = tst_simple

# Test output to be deleted.
#CLEANFILES = ???
