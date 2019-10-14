program threadohd
  use omp_lib
  use gptl
  implicit none

  integer :: n, ret, iter

  ret = gptlsetutr (gptlnanotime)
  ret = gptlinitialize ()

  do iter=1,10
    ret = gptlstart_threadohd_outer ('loop')
!$OMP PARALLEL DO PRIVATE (ret) SCHEDULE (runtime)
    do n=1,1000
      ret = gptlstart_threadohd_inner ('loop')
      ret = gptlstart ('base')
      call sub ()
      ret = gptlstop ('base')
      ret = gptlstop_threadohd_inner ('loop')
    end do
    ret = gptlstop_threadohd_outer ('loop')
  end do

  ret = gptlpr (0)
end program threadohd

subroutine sub()
  use omp_lib
  implicit none

  integer :: mythread

  mythread = omp_get_thread_num ()
end subroutine sub

