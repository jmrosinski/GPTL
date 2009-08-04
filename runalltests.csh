#!/bin/csh -f

# Test the GPTL library by permuting the default values of as many user-settable
# parameters as possible. The list defined by the "foreach" loop below will 
# need to be culled of all tests which truly can't be changed. For example if
# PAPI is unavailable, delete HAVE_PAPI from the list because this script will
# try to run a PAPI test where HAVE_PAPI is permuted from "no" to "yes", and
# the test will fail.

set basescript = macros.make.linux  # This is the base script to start from
echo "$0 Testing $basescript..."
cp $basescript macros.make || echo "Failure to cp $basescript to macros.make" && exit 1
make clean; make || echo "Failure in make" && exit 1
make test  || echo "Failure in make test" && exit 1
echo "$0 $basescript worked" >! results

# Will need to delete from user settable list all items which truly aren't available
foreach usersettable ( PTHREADS )
#foreach usersettable ( DEBUG OPENMP PTHREADS FORTRAN HAVE_PAPI HAVE_MPI TEST_AUTOPROFILE )
grep "^ *$usersettable *= *yes *" $basescript
set ret = $status

# Determine whether to toggle from no to yes, or yes to no
if ($ret == 0) then
  set oldtest = yes
  set newtest = no
else
  set oldtest = no
  set newtest = yes
endif
echo "$0 Testing $usersettable = $newtest ..."
echo "$0 Testing $usersettable = $newtest ..." >> results

# For PTHREADS case, ensure OPENMP is off
if ( $usersettable == PTHREADS ) then
  sed -e "s/^ *OPENMP *= *yes */OPENMP = no/g" \
      -e "s/^ *$usersettable *= *$oldtest */$usersettable = $newtest/g" $basescript >! macros.make
else
  sed -e "s/^ *$usersettable *= *$oldtest */$usersettable = $newtest/g" $basescript >! macros.make
endif

make clean; make || echo "Failure in make" && exit 1
make test  || echo "Failure in make test" && exit 1
echo "$usersettable = $newtest worked" >> results
end

echo "Permuting all user settable tests passed" && exit 0
