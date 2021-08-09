program utrtest
  use gptl
  use gptl_acc

  implicit none

  logical :: enable_expensive = .false. ! true means order calls with collisions badly
  double precision :: sum
  integer :: ret
  integer :: handle1, handle2, handle3, handle4, handle5, handle6, handle7, handle8
  integer :: handle1_gpu, handle2_gpu, handle3_gpu, handle4_gpu, handle5_gpu, handle6_gpu, &
             handle7_gpu, handle8_gpu, handle_total
  integer :: khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu
  integer :: n                 ! iterator
  integer :: narg              ! number of cmd-line args
  character(len=256) :: arg    ! cmd-line arg

!$acc routine(sub_gpu) seq
  external :: sub_gpu
  
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

  write(6,*) 'Purpose: assess accuracy of GPTL overhead estimates'
  
! Retrieve information about the GPU. Need only cores_per_gpu
  ret = gptlget_gpu_props (khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu)
  ret = gptlinitialize ()

! Set CPU handles
  ret = gptlinit_handle ('1e7x1',handle8)
  ret = gptlinit_handle ('1e6x10',handle7)
  ret = gptlinit_handle ('1e5x100',handle6)
  ret = gptlinit_handle ('1e4x1000',handle5)
  ret = gptlinit_handle ('1000x1e4',handle4)
  ret = gptlinit_handle ('100x1e5',handle3)
  ret = gptlinit_handle ('10x1e6',handle2)
  ret = gptlinit_handle ('1x1e7',handle1)

! Set GPU handles
!$acc parallel private(ret) &
!$acc&  copyout(handle8_gpu,handle7_gpu,handle6_gpu,handle5_gpu,handle4_gpu, &
!$acc&          handle3_gpu,handle2_gpu,handle1_gpu,handle_total)
  ret = gptlinit_handle_gpu ('total', handle_total);
  ret = gptlinit_handle_gpu ('1e7x1',handle8_gpu)
  ret = gptlinit_handle_gpu ('1e6x10',handle7_gpu)
  ret = gptlinit_handle_gpu ('1e5x100',handle6_gpu)
  ret = gptlinit_handle_gpu ('1e4x1000',handle5_gpu)
  ret = gptlinit_handle_gpu ('1000x1e4',handle4_gpu)
  ret = gptlinit_handle_gpu ('100x1e5',handle3_gpu)
  ret = gptlinit_handle_gpu ('10x1e6',handle2_gpu)
  ret = gptlinit_handle_gpu ('1x1e7',handle1_gpu)
!$acc end parallel
  ret = gptlcudadevsync ()

  sum = 0.
  ret = gptlstart ('total')
  ret = gptlstart ('total_cpu')
  call sub (10000000, 1, sum, '1e7x1'//char(0), handle8)
  call sub (1000000, 10, sum, '1e6x10'//char(0), handle7)
  call sub (100000, 100, sum, '1e5x100'//char(0), handle6)
  call sub (10000, 1000, sum, '1e4x1000'//char(0), handle5)
  call sub (1000, 10000, sum, '1000x1e4'//char(0), handle4)
  call sub (100, 100000, sum, '100x1e5'//char(0), handle3)
  call sub (10, 1000000, sum, '10x1e6'//char(0), handle2)
  call sub (1, 10000000, sum, '1x1e7'//char(0), handle1)
  ret = gptlstop ('total_cpu')

  sum = 0.
!$acc parallel private (ret,n) &
!$acc  copyin(handle1_gpu,handle2_gpu,handle3_gpu,handle4_gpu,handle5_gpu,    &
!$acc	      handle6_gpu,handle7_gpu,handle8_gpu,handle_total,cores_per_gpu) &
!$acc  copy(sum) reduction(sum)
!$acc loop gang worker vector
  do n=1,cores_per_gpu
    ret = gptlstart_gpu (handle_total)
    call sub_gpu (10000000, 1, sum, handle8_gpu)
    call sub_gpu (1000000, 10, sum, handle7_gpu)
    call sub_gpu (100000, 100, sum, handle6_gpu)
    call sub_gpu (10000, 1000, sum, handle5_gpu)
    call sub_gpu (1000, 10000, sum, handle4_gpu)
    call sub_gpu (100, 100000, sum, handle3_gpu)
    call sub_gpu (10, 1000000, sum, handle2_gpu)
    call sub_gpu (1, 10000000, sum, handle1_gpu)
    ret = gptlstop_gpu (handle_total)
  end do
!$acc end parallel
  ret = gptlcudadevsync ()
  ret = gptlstop ('total')
    
  ret = gptlpr (-1)  ! negative number means write to stderr
  ret = gptlcudadevsync ()
  write(6,*) 'Final value of sum=',sum
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

subroutine sub (outer, inner, sum, name, handle)
  use gptl
  implicit none
  integer, intent(in) :: outer, inner
  double precision, intent(inout) :: sum
  character(len=*), intent(in) :: name
  integer, intent(inout) :: handle
  integer :: i, j, ret

  do i=0,outer-1
    ret = gptlstart_handle (name, handle)
    do j=0,inner-1
      sum = sum + j
    end do
    ret = gptlstop_handle (name, handle)
  end do
end subroutine sub

subroutine sub_gpu (outer, inner, sum, handle)
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
end subroutine sub_gpu
