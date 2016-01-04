program threadohd
  use mpi
  use gptl
  implicit none

  integer :: n, ret, iter
  integer :: ierr, myrank, nranks
  real :: arr(48,48)
  integer, external :: omp_get_max_threads

  call mpi_init (ierr)
  call mpi_comm_rank (MPI_COMM_WORLD, myrank, ierr)
  call mpi_comm_size (MPI_COMM_WORLD, nranks, ierr)

  ret = gptlsetutr (gptlnanotime)
  ret = gptlinitialize ()

  do iter=1,10
    ret = gptlstart_threadohd_outer ('loop')
!$OMP PARALLEL DO PRIVATE (ret) SCHEDULE (runtime)
    do n=1,1000
      ret = gptlstart_threadohd_inner ('loop')
      ret = gptlstart ('base')
      call sub (myrank)
      ret = gptlstop ('base')
      ret = gptlstop_threadohd_inner ('loop')
    end do
    ret = gptlstop_threadohd_outer ('loop')
  end do

  ret = gptlpr (myrank)
  ret = gptlpr_summary (MPI_COMM_WORLD)
  call mpi_finalize (ierr)
end program threadohd

subroutine sub(myrank)
  implicit none
  integer, intent(in) :: myrank

  integer :: mythread
  integer, external :: omp_get_thread_num

  mythread = omp_get_thread_num ()
! No standard way to sleep in Fortran so comment this out
!  call msleep(myrank+mythread+1)
end subroutine sub

