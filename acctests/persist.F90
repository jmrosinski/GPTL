subroutine persist (myrank, mostwork, maxthreads_gpu, outerlooplen, innerlooplen, balfact)
  use gptl
  use gptl_acc
  implicit none
!$acc routine (doalot_log) seq
!$acc routine (doalot_log_inner) seq
!$acc routine (doalot_sqrt) seq

  integer, intent(in) :: myrank
  integer, intent(in) :: mostwork
  integer, intent(in) :: maxthreads_gpu
  integer, intent(in) :: outerlooplen
  integer, intent(in) :: innerlooplen
  integer, intent(in) :: balfact

  integer :: ret
  integer :: n
  integer :: niter

  integer :: handle = 0, handle2 = 0
  real :: factor

  real :: vals(outerlooplen)
  real, external :: doalot_log, doalot_log_inner, doalot_sqrt

!JR NOTE: gptlinitialize call increases mallocable memory size on GPU. That call will fail
!JR if any GPU activity happens before the call to gptlinitialize
  ret = gptlsetoption (gptlmaxthreads_gpu, maxthreads_gpu)
!  ret = gptlsetoption (gptltablesize_gpu, 32)   ! This setting gives 1 collision
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

  ret = gptlstart ('all_gpuloops')
  ret = gptlstart ('do_nothing_cpu')
!$acc parallel loop copyin(balfact,handle,handle2) copyout(ret)
  do n=0,outerlooplen-1
    ret = gptlstart_gpu ('all_gpucalls')
    ret = gptlstart_gpu ('do_nothing_gpu')
    ret = gptlstop_gpu ('do_nothing_gpu')
    ret = gptlstop_gpu ('all_gpucalls')
  end do
!$acc end parallel
  ret = gptlstop ('do_nothing_cpu')

  ret = gptlstart ('doalot_cpu')
!$acc parallel loop private(niter,factor) copyin(balfact,handle,handle2) copyout(ret, vals)
  do n=0,outerlooplen-1
    ret = gptlstart_gpu ('all_gpucalls')
    factor = real(n) / real(outerlooplen-1)
    select case (balfact)
    case (0)
      niter = int(factor * mostwork)
    case (1)
      niter = mostwork
    case (2)
      niter = mostwork - int(factor * mostwork)
    end select

    ret = gptlstart_gpu ('doalot_log')
    vals(n) = doalot_log (niter, innerlooplen)
    ret = gptlstop_gpu ('doalot_log')

    ret = gptlstart_gpu ('doalot_log_inner')
    vals(n) = doalot_log_inner (niter, innerlooplen)
    ret = gptlstop_gpu ('doalot_log_inner')

    ret = gptlstart_gpu ('doalot_sqrt')
    vals(n) = doalot_sqrt (niter, innerlooplen)
    ret = gptlstop_gpu ('doalot_sqrt')

    ret = gptlstart_gpu_c ('doalot_sqrt_c'//char(0))
    vals(n) = doalot_sqrt (niter, innerlooplen)
    ret = gptlstop_gpu_c ('doalot_sqrt_c'//char(0))

    ret = gptlstart_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)
    vals(n) = doalot_sqrt (niter, innerlooplen)
    ret = gptlstop_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)

    ret = gptlstart_handle_gpu_c ('a'//char(0), handle2)
    vals(n) = doalot_sqrt (niter, innerlooplen)
    ret = gptlstop_handle_gpu_c ('a'//char(0), handle2)
    ret = gptlstop_gpu ('all_gpucalls')
  end do
!$acc end parallel
  ret = gptlstop ('doalot_cpu')
  ret = gptlstop ('all_gpuloops')
  
  ret = gptlstart ('doalot_cpu_nogputimers')
!$acc parallel loop private(niter,factor) copyin(balfact) copyout(vals)
  do n=0,outerlooplen-1
    factor = real(n) / real(outerlooplen-1)
    select case (balfact)
    case (0)
      niter = int(factor * mostwork)
    case (1)
      niter = mostwork
    case (2)
      niter = mostwork - int(factor * mostwork)
    end select
    
    vals(n) = doalot_log (niter, innerlooplen)
    vals(n) = doalot_sqrt (niter, innerlooplen)
    vals(n) = doalot_sqrt (niter, innerlooplen)
    vals(n) = doalot_sqrt (niter, innerlooplen)
    vals(n) = doalot_sqrt (niter, innerlooplen)
  end do
!$acc end parallel
  ret = gptlstop ('doalot_cpu_nogputimers')

  write(6,*)'Sleeping 1 second on GPU...'
  ret = gptlstart ('all_gpuloops')
  ret = gptlstart ('sleep1ongpu')
!$acc parallel loop private(ret)
  do n=1,outerlooplen
    ret = gptlstart_gpu ('all_gpucalls')
    ret = gptlstart_gpu ('sleep1')
    ret = gptlmy_sleep (1.)
    ret = gptlstop_gpu ('sleep1')
    ret = gptlstop_gpu ('all_gpucalls')
  end do
!$acc end parallel

  ret = gptlstop ('sleep1ongpu')
  ret = gptlstop ('all_gpuloops')

  ret = gptlpr (myrank)
end subroutine persist

real function doalot_log (n, innerlooplen) result (sum)
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
end function doalot_log

! doalot_log_inner: Same computations as doalot_log, but add a timer inside "innerlooplen"
real function doalot_log_inner (n, innerlooplen) result (sum)
  use gptl_acc
  implicit none
  integer, intent(in) :: n, innerlooplen
  integer :: i, iter
  integer :: ret
  real :: sum
!$acc routine seq

  sum = 0.
  do iter=1,innerlooplen
    ret = gptlstart_gpu ('doalot_log_inner_iter')
    do i=1,n
      sum = sum + log (real (iter*i))
    end do
    ret = gptlstop_gpu ('doalot_log_inner_iter')
  end do
end function doalot_log_inner

real function doalot_sqrt (n, innerlooplen) result (sum)
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
end function doalot_sqrt

