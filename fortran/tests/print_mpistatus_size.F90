program print_mpistatus_size
  use mpi
  implicit none
  write(6,*) 'MPI_STATUS_SIZE=', MPI_STATUS_SIZE
end program print_mpistatus_size
