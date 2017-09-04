program simple
  use gptl
  implicit none
#include "../cuda/gptl.inc"

  integer :: ret
  integer :: n
  integer, external :: dummy

!$acc routine (sub) seq

  write(6,*)'simple: calling gptlinitialize 1'
  ret = gptlinitialize ()
  write(6,*)'simple: calling dummy'
  ret = dummy ()

  ret = gptlstart ('total')
  write(6,*) 'Entering kernels loop'
  call flush(6)
!$acc kernels
  do n=1,1
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

integer function dummy ()
  implicit none
#include "../cuda/gptl.inc"

  integer :: ret
!$acc routine (gptldummy_gpu) seq

  write(6,*)'entered dummy'
  ret = gptldummy_gpu ()
end function dummy
