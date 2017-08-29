program simple
  use gptl
  use gptl_gpu
  implicit none

  integer :: ret
  integer :: n

!$acc routine (sub) seq

  ret = gptlinitialize ()
  ret = gptlstart ('total')
  write(6,*) 'Entering kernels loop'
  call flush(6)
!$acc kernels
  do n=1,1000
    call sub ()
  end do
!$acc end kernels
  write(6,*) 'Exiting kernels loop'
  call flush(6)
  ret = gptlstop ('total')
  write(6,*) 'Calling gptlpr'
  call flush(6)
  ret = gptlpr (0)
end program simple

subroutine sub ()
  use gptl_gpu
  implicit none

  integer :: ret
!$acc routine seq

  ret = gptlstart_gpu ('sub')
  ret = gptlstart_gpu ('innersub')
  ret = gptlstop_gpu ('innersub')
  ret = gptlstop_gpu ('sub')
end subroutine sub
