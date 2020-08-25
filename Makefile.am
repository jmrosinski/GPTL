## This is a automake file, part of the GPTL package.
# Copyright 2018, see the COPYING file for more information.

# This is the main automake file for GPTL.
# Ed Hartnett, 5/18/18

# This directory stores libtool macros, put there by aclocal.
ACLOCAL_AMFLAGS = -I m4

# These files get added to the distribution.
EXTRA_DIST = COPYING README INSTALL

# This is the list of subdirs for which Makefiles will be constructed
SUBDIRS = include src 

# Need to build cuda lib before tests
if ENABLE_CUDA
SUBDIRS += cuda
endif

SUBDIRS += tests bin man

if HAVE_FORTRAN
SUBDIRS += fortran
endif

if ENABLE_FACC
SUBDIRS += facc
endif

if ENABLE_CACC
SUBDIRS += cacc
endif

# Name and dir to hold pkgconfig info
pkgconfigdir = ${libdir}/pkgconfig
pkgconfig_DATA = gptl.pc

install-data-hook:

	@echo ''
	@echo '+-------------------------------------------------------------+'
	@echo '| GPTL installation successful                                |'
	@echo '|                                                             |'
	@echo '| CAUTION:                                                    |'
	@echo '|                                                             |'
	@echo '| If you have not already run "make check", then we strongly  |'
	@echo '| recommend you do so. It does not take very long.            |'
	@echo '|                                                             |'
	@echo '| GPTL was developed with support from multiple institutions, |'
	@echo '| including NOAA, NCAR, and ORNL.                             |'
	@echo '+-------------------------------------------------------------+'
	@echo ''
