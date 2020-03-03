program omptest
  use gptl
  use omp_lib

  implicit none
  external :: sub
  
  integer :: ret
  integer :: iter
  real*8 val
  character(len=*), parameter :: thisprog = "omptest"

  call omp_set_num_threads (2)
  ret = gptlinitialize ()
  ret = gptlstart ("main")
  ret = gptlstart ("omp_loop")
  !$OMP PARALLEL DO PRIVATE (ITER)
  do iter=0,1
    call sub (iter)
  end do
  ret = gptlstop ("omp_loop");
  ret = gptlstop ("main");

  ! This test should succeed
  ret = GPTLget_wallclock ("sub", 1, val)
  if (ret /= 0) then
    write(6,*) thisprog,": GPTLget_wallclock failure for thread 1"
    call exit (1)
  end if

  ! This test should fail
  ret = GPTLget_wallclock ("sub", 2, val)
  if (ret == 0) then
    write(6,*) thisprog,": GPTLget_wallclock should have failed for thread 2"
    call exit (1)
  end if
  stop 0
end program omptest

subroutine sub (iter)
  integer, intent(in) :: iter
  integer ret;
  integer mythread

  mythread = omp_get_thread_num ()
  ret = GPTLstart ("sub");
  write(6,*) "iter=",iter," being processed by thread=",mythread
  ret = GPTLstop ("sub");
end subroutine sub
