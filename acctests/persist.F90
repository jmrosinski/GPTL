program persist
  use gptl
  use gptl_acc
  implicit none
!$acc routine (doalot) seq
!$acc routine (doalot2) seq

  integer :: ret
  integer :: n
  integer :: maxthreads_gpu = 3584
  integer :: outerlooplen
  integer :: innerlooplen = 100
  integer :: balfact = 1
  integer :: niter
  integer :: ans
  integer :: handle = 0, handle2 = 0
  real :: factor = 1.

  real, allocatable :: vals(:)
  real, external :: doalot, doalot2

  call getval (maxthreads_gpu, 'maxthreads_gpu')
  call getval_real (factor, 'factor: maxthreads * FACTOR = outerlooplen')
  outerlooplen = maxthreads_gpu * nint (factor)
  write(6,*)'outerlooplen=',outerlooplen
  call getval (innerlooplen, 'innerlooplen')
  call getval (balfact, 'balfact: 0=LtoR 1=balanced 2=RtoL')
  allocate (vals(outerlooplen))
!JR NOTE: gptlinitialize call increases mallocable memory size on GPU. That call will fail
!JR if any GPU activity happens before the call to gptlinitialize
  ret = gptlsetoption (gptlmaxthreads_gpu, maxthreads_gpu)
  write(6,*)'persist: calling gptlinitialize'
  ret = gptlinitialize ()
!$acc kernels
  call gptldummy_gpu (0)
!$acc end kernels
  
!JR Need to call GPU-specific init_handle routine because its tablesize may differ from CPU
!$acc parallel copyout(ret,handle,handle2)
  call gptldummy_gpu (1)
  ret = gptlinit_handle_gpu ('doalot_handle_sqrt_c', handle)
  call gptldummy_gpu (2)
  ret = gptlinit_handle_gpu ('a', handle2)
!$acc end parallel

  ret = gptlstart ('doalot_cpu')
!$acc parallel loop private(niter) copyin(balfact,handle,handle2) copyout(ret, vals)
  do n=1,outerlooplen
    select case (balfact)
    case (0)
      niter = n
    case (1)
      niter = outerlooplen
    case (2)
      niter = outerlooplen - n + 1
    end select
    
    ret = gptlstart_gpu ('doalot_log')
    vals(n) = doalot (niter, innerlooplen)
    ret = gptlstop_gpu ('doalot_log')

    ret = gptlstart_gpu ('doalot_sqrt')
    vals(n) = doalot2 (niter, innerlooplen)
    ret = gptlstop_gpu ('doalot_sqrt')

    ret = gptlstart_gpu_c ('doalot_sqrt_c'//char(0))
    vals(n) = doalot2 (niter, innerlooplen)
    ret = gptlstop_gpu_c ('doalot_sqrt_c'//char(0))

    ret = gptlstart_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)
    vals(n) = doalot2 (niter, innerlooplen)
    ret = gptlstop_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)

    ret = gptlstart_handle_gpu_c ('a'//char(0), handle2)
    vals(n) = doalot2 (niter, innerlooplen)
    ret = gptlstop_handle_gpu_c ('a'//char(0), handle2)
  end do
!$acc end parallel
  ret = gptlstop ('doalot_cpu')

  ret = gptlstart ('doalot_cpu_nogputimers')

!$acc parallel loop private(niter) copyin(balfact) copyout(vals)
  do n=1,outerlooplen
    select case (balfact)
    case (0)
      niter = n
    case (1)
      niter = outerlooplen
    case (2)
      niter = outerlooplen - n + 1
    end select
    vals(n) = doalot (niter, innerlooplen)
    vals(n) = doalot2 (niter, innerlooplen)
    vals(n) = doalot2 (niter, innerlooplen)
    vals(n) = doalot2 (niter, innerlooplen)
    vals(n) = doalot2 (niter, innerlooplen)
  end do
!$acc end parallel
  ret = gptlstop ('doalot_cpu_nogputimers')
  ret = gptlpr (0)
end program persist

real function doalot (n, innerlooplen) result (sum)
  implicit none
  integer, intent(in) :: n, innerlooplen
  integer :: i, iter
  real :: sum
!$acc routine seq

  sum = 0.
  do iter=1,innerlooplen
    do i=1,n
      sum = sum + log (real (iter*i))
    end do
  end do
end function doalot

real function doalot2 (n, innerlooplen) result (sum)
  implicit none
  integer, intent(in) :: n, innerlooplen
  integer :: i, iter
  real :: sum
!$acc routine seq

  sum = 0.
  do iter=1,innerlooplen
    do i=1,n
      sum = sum + sqrt (real (iter*i))
    end do
  end do
end function doalot2

subroutine getval (arg, str)
  implicit none

  integer, intent(inout) :: arg
  character(len=*), intent(in) :: str

  integer :: ans

  write(6,'(a,a,a,i9,a)')'Enter ',str,' or -1 to accept default (',arg,')'
  read(5,*) ans
  if (ans /= -1) then
    arg = ans
  end if
  write(6,*) 'arg=',arg
end subroutine getval

subroutine getval_real (arg, str)
  implicit none

  real, intent(inout) :: arg
  character(len=*), intent(in) :: str

  real :: ans

  write(6,'(a,a,a,f5.3,a)')'Enter ',str,' or -1 to accept default (',arg,')'
  read(5,*) ans
  if (ans /= -1.) then
    arg = ans
  end if
  write(6,*) 'arg=',arg
end subroutine getval_real
