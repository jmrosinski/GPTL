program inplace
  ! Driver for tests of profiled routines making use of MPI_IN_PLACE. These need to be in
  ! Fortran because unfortunately many if not all MPI implementations put the Fortran test
  ! for MPI_IN_PLACE in their Fortran wrapper which is written in C. This means if GPTL
  ! profiling is enabled, the test for MPI_IN_PLACE needs to be in the (GPTL) wrapper code as
  ! well. In other words, passing MPI_IN_PLACE from Fortran needs to be handled in the
  ! wrapper or wrong answers will result.
  !
  ! To make the tests fail, initialize f_mpi_in_place to 0 in fortran/src/f_wrappers_pmpi.c 
  use mpi
  use gptl
  use inplace_subs

  implicit none
  integer :: myid, numprocs, rc, ierr
  integer :: ret
  integer :: nbad = 0
  
  call MPI_INIT( ierr )
  call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
  call MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )

  ret = gptlinitialize ()

  nbad = nbad + reduce     (myid, numprocs)
  nbad = nbad + allreduce  (myid, numprocs)
  nbad = nbad + gather     (myid, numprocs)
  nbad = nbad + gatherv    (myid, numprocs)
  nbad = nbad + scatter    (myid, numprocs)
  nbad = nbad + scatterv   (myid, numprocs)
  nbad = nbad + allgather  (myid, numprocs)
  nbad = nbad + allgatherv (myid, numprocs)
  ! Invoking either of these should result in MPI_Abort() from "make check" since
  ! a check has been added to f_wrappers_pmpi.c to disallow MPI_IN_PLACE
  if (.false.) then
    nbad = nbad + alltoall (myid, numprocs)
    nbad = nbad + alltoallv (myid, numprocs)
  end if

  ! If any functions produced wrong answers, abort so script knows
  if (nbad > 0) then
    write(6,*)'inplace: a total of ', nbad, ' bad return codes occurred'
    call MPI_Abort (MPI_COMM_WORLD, -1, ierr)
  end if

  call MPI_FINALIZE(rc)
  stop 0
end program inplace
