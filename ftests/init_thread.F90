program init_thread
  use gptl
  implicit none

  integer :: ret
  integer :: n, nthreads
  real*8 :: init_time, get_time
  integer, external :: omp_get_max_threads

  nthreads = omp_get_max_threads ()

  ret = gptlsetutr (gptlnanotime)
  ret = gptlinitialize ()
!$OMP PARALLEL DO PRIVATE (ret)
  do n=1,nthreads
    ret = gptlstart ('get_thread')
    ret = gptlstop ('get_thread')
  end do

!$OMP PARALLEL DO PRIVATE (ret)
  do n=1,nthreads
    ret = gptlstart ('get_thread')
    ret = gptlstop ('get_thread')
  end do

  ret = gptlpr (0)
  write(6,*)'nthreads, get time (sec)=',nthreads, get_time
end program init_thread
