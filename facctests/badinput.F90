program basic
  use gptl
  use gptl_acc
  
  implicit none

  character(len=*), parameter :: prog='badinput'
  integer :: ret, ret2
  integer :: handle1, handle2
  integer :: handle3 = -1

  ! Only allow 2 timers
  ret = gptlsetoption (gptlmaxtimers_gpu, 2)

#ifdef ENABLE_GPUCHECKS
  write(6,*) 'Testing gptlinit_handle_gpu before gptlinitialize'
!$acc parallel copyout(ret) copy(handle1)
  ret = gptlinit_handle_gpu ('badinit', handle1)
!$acc end parallel
  ret2 = gptlcudadevsync ()
  if (ret == 0) then
    write (6,*) 'Expected not initialized from gptlinit_handle_gpu got 0'
    stop 1
  end if

! When DUMMYGPUSTARTSTOP is set, gptlstart_gpu and gtlstop_gpu just returns 0
#ifndef DUMMYGPUSTARTSTOP
  write(6,*) 'Testing gptlstart_gpu before gptlinitialize'
!$acc parallel copyin(handle1) copyout(ret)
  ret = gptlstart_gpu (handle1)
!$acc end parallel
  ret2 = gptlcudadevsync ()
  if (ret == 0) then
    write (6,*) 'Expected not initialized from gptlstart_gpu got 0'
    stop 2
  end if
    
  write(6,*) 'Testing gptlstop_gpu before gptlinitialize'
!$acc parallel copyin(handle1) copyout(ret)
  ret = gptlstop_gpu (handle1)
!$acc end parallel
  ret2 = gptlcudadevsync ()
  if (ret == 0) then
    write (6,*) 'Expected not initialized from gptlstop_gpu got 0'
    stop 3
  end if
#endif
  
  ! Initialize the GPTL library on CPU and GPU
  ret = gptlinitialize ()

  write(6,*) 'Testing gptlinit_handle_gpu for too many timers'
!$acc parallel copyout(ret) copy(handle1,handle2,handle3)
  ret = gptlinit_handle_gpu ('timer1', handle1)
  ret = gptlinit_handle_gpu ('timer2', handle2)
  ret = gptlinit_handle_gpu ('timer3', handle3)
!$acc end parallel
  ret2 = gptlcudadevsync ()
  if (ret == 0) then
    write (6,*) 'Expected bad handle (too many timers) from gptlinit_handle_gpu got 0'
    stop 4
  end if

! When DUMMYGPUSTARTSTOP is set, gptlstart_gpu and gtlstop_gpu just returns 0
#ifndef DUMMYGPUSTARTSTOP
  write(6,*) 'Testing gptlstart_gpu for bad handle'
!$acc parallel copyin(handle3) copyout(ret)
  ret = gptlstart_gpu (handle3)
!$acc end parallel
  ret2 = gptlcudadevsync ()
  if (ret == 0) then
    write (6,*) 'Expected bad handle (too many timers) from gptlstart_gpu got 0'
    stop 5
  end if

  write(6,*) 'Testing gptlstop_gpu for bad handle'
!$acc parallel copyin(handle3) copyout(ret)
  ret = gptlstop_gpu (handle3)
!$acc end parallel
  ret2 = gptlcudadevsync ()
  if (ret == 0) then
    write (6,*) 'Expected bad handle (too many timers) from gptlstop_gpu got 0'
    stop 6
  end if

  write(6,*) 'Testing gptlstop_gpu for timer already off'
!$acc parallel copyin(handle1) copyout(ret)
  ret = gptlstart_gpu (handle1)
  ret = gptlstop_gpu (handle1)
  ret = gptlstop_gpu (handle1)
!$acc end parallel
  ret2 = gptlcudadevsync ()
  if (ret == 0) then
    write (6,*) 'Expected bad return from gptlstop_gpu (timer already off) got 0'
    stop 7
  end if
#endif

#else
  write (6,*) 'Doing nothing since ENABLE_GPUCHECKS is not defined'
#endif
  stop 0
end program basic
