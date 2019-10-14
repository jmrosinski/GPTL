program printmpistatussize
#ifdef HAVE_LIBMPI
  use mpi
#endif
  implicit none
#ifdef HAVE_LIBMPI
  write(6,*) 'MPI_STATUS_SIZE=', MPI_STATUS_SIZE
#else
  write(6,*) 'Need to run configure with something like env FC=mpif90 to get MPI_STATUS_SIZE'
#endif
end program printmpistatussize
