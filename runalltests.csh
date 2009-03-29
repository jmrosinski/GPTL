#!/bin/csh -f

echo "$0 Testing default macros.make.linux..."
cp macros.make.base macros.make
make clean; make || (echo Failure && exit 1)
make test  || (echo Failure && exit 1)

echo "$0 Testing DEBUG = yes..."
sed -e 's/^ *DEBUG *= *no *$/DEBUG = yes/1' macros.make.base >! macros.make
make clean; make || (echo Failure && exit 1)
make test  || (echo Failure && exit 1)

echo "$0 Testing OPENMP = no..."
sed -e 's/^ *OPENMP *= *yes *$/OPENMP = no/1' macros.make.base >! macros.make
make clean; make || (echo Failure && exit 1)
make test  || (echo Failure && exit 1)

echo "$0 Testing no Fortran..."
sed -e 's/^ *FORTRAN *= *yes *$/FORTRAN = no/1' macros.make.base >! macros.make
make clean; make || (echo Failure && exit 1)
make test  || (echo Failure && exit 1)

echo "$0 Testing yes PAPI..."
sed -e 's/^ *HAVE_PAPI *= *no *$/HAVE_PAPI = yes/1' macros.make.base >! macros.make
make clean; make || (echo Failure && exit 1)
make test  || (echo Failure && exit 1)

echo "$0 Testing HAVE_MPI = yes..."
sed -e 's/^ *HAVE_MPI *= *no *$/HAVE_MPI = yes/1' macros.make.base >! macros.make
make clean; make || (echo Failure && exit 1)
make test  || (echo Failure && exit 1)

echo "$0 Testing TEST_AUTOPROFILE = no..."
sed -e 's/^ *TEST_AUTOPROFILE *= *yes *$/TEST_AUTOPROFILE = no/1' macros.make.base >! macros.make
make clean; make || (echo Failure && exit 1)
make test  || (echo Failure && exit 1)
