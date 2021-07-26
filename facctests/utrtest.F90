program utrtest
  use gptl
  use gptl_acc

  implicit none

  logical :: enable_expensive = .false. ! true means order calls with collisions badly
  double precision :: sum = 0.
  integer :: ret
  integer :: handle1, handle2, handle3, handle4, handle5, handle6, handle7, handle8

  integer :: n                 ! iterator through argument list
  integer :: narg              ! number of cmd-line args
  character(len=256) :: arg    ! cmd-line arg

!$acc routine(sub) seq
  external :: sub
  
  narg = command_argument_count ()
  n = 1
  do while (n <= narg)
    call get_command_argument (n, arg)
    if (trim(arg) == '-e') then
      enable_expensive = .false.
      n = n + 1
    else
      write(6,*)'Unknown flag ',trim(arg),' Only -e is known'
      stop 1
    end if
  end do

  write(6,*) 'Purpose: estimate overhead of GPTL underlying timing routine (UTR)'
  
  ret = gptlinitialize ()

! Set handles outside of threaded loop
!$acc parallel private(ret) copyout(handle8,handle7,handle6,handle5,handle4,handle3,handle2,handle1)
  ret = gptlinit_handle_gpu ('1e7x1', handle8)
  ret = gptlinit_handle_gpu ('1e6x10', handle7)
  ret = gptlinit_handle_gpu ('1e5x100', handle6)
  ret = gptlinit_handle_gpu ('1e4x1000', handle5)
  ret = gptlinit_handle_gpu ('1000x1e4', handle4)
  ret = gptlinit_handle_gpu ('100x1e5', handle3)
  ret = gptlinit_handle_gpu ('10x1e6', handle2)
  ret = gptlinit_handle_gpu ('1x1e7', handle1)
!$acc end parallel
  ret = gptlcudadevsync ()
  
  ret = gptlstart ('total')
  if (enable_expensive) then
!$acc parallel copyin(handle1,handle2,handle3,handle4,handle5,handle6,handle7,handle8,sum)
    call sub (1, 10000000, sum, handle1)
    call sub (10, 1000000, sum, handle2)   ! collides
    call sub (100, 100000, sum, handle3)
    call sub (1000, 10000, sum, handle4)
    call sub (10000, 1000, sum, handle5)
    call sub (100000, 100, sum, handle6)
    call sub (1000000, 10, sum, handle7)   ! collides
    call sub (10000000, 1, sum, handle8)
!$acc end parallel
  else
!$acc parallel copyin(handle1,handle2,handle3,handle4,handle5,handle6,handle7,handle8,sum)
    call sub (10000000, 1, sum, handle8)
    call sub (1000000, 10, sum, handle7)
    call sub (100000, 100, sum, handle6)
    call sub (10000, 1000, sum, handle5)
    call sub (1000, 10000, sum, handle4)
    call sub (100, 100000, sum, handle3)
    call sub (10, 1000000, sum, handle2)
    call sub (1, 10000000, sum, handle1)
!$acc end parallel
  end if
  ret = gptlcudadevsync ()
  ret = gptlstop ('total')
    
  ret = gptlpr (-1)  ! negative number means write to stderr
  stop 0

CONTAINS

  subroutine printusemsg_exit
    implicit none
    integer :: n
    write(6,*)'Usage: utrtest [-e]'
    write(6,*)'where -e enables expensive sequencing'
    stop 1
  end subroutine printusemsg_exit
end program utrtest

subroutine sub (outer, inner, sum, handle)
  use gptl_acc
  implicit none
  integer, intent(in) :: outer, inner
  double precision, intent(inout) :: sum
  integer, intent(inout) :: handle
  integer :: i, j, ret

!$acc routine seq                                                               
  do i=0,outer-1
    ret = gptlstart_gpu (handle)
    do j=0,inner-1
      sum = sum + j
    end do
    ret = gptlstop_gpu (handle)
  end do
end subroutine sub
