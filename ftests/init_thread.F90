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
  ret = gptlstart ('gptlinit_thread')
!$OMP PARALLEL DO PRIVATE (ret)
  do n=1,nthreads
    ret = gptlinit_thread ()
  end do
  ret = gptlstop ('gptlinit_thread')

  ret = gptlstart ('get_thread')
!$OMP PARALLEL DO PRIVATE (ret)
  do n=1,nthreads
    ret = gptlinit_thread ()
  end do
  ret = gptlstop ('get_thread')

  ret = gptlget_wallclock ('gptlinit_thread', 0, init_time)
  ret = gptlget_wallclock ('get_thread', 0, get_time)
  write(6,*)'nthreads, init time (sec)=',nthreads, init_time
  write(6,*)'nthreads, get time (sec)=',nthreads, get_time
end program init_thread
