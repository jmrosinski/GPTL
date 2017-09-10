program persist
  use gptl
  use gptl_acc
  implicit none
!$acc routine (gptlinit_handle_gpu) seq
!$acc routine (doalot) seq
!$acc routine (doalot2) seq

  integer :: ret
  integer :: n
  integer :: pret(1000)
  integer :: handle, handle2
  real :: vals(1000)
!  integer, external :: gptlstart_gpu, gptlstop_gpu
  real, external :: doalot, doalot2

  write(6,*)'persist: calling gptlinitialize'
!JR NOTE: gptlinitialize call increases mallocable memory size on GPU. That call will fail
!JR if any GPU activity happens before the call to gptlinitialize
  ret = gptlsetoption (gptlmaxthreads_gpu, 3584)
  ret = gptlinitialize ()
!JR Need to call GPU-specific init_handle routine because its tablesize may differ from CPU
!$acc kernels copyout(ret,handle,handle2)
  ret = gptlinit_handle_gpu ('doalot_handle_sqrt_c', handle)
  ret = gptlinit_handle_gpu ('a', handle2)
!$acc end kernels

  ret = gptlstart ('doalot_cpu')
!$acc parallel loop copyin(handle) copyout(ret, vals)
  do n=1,1000
    ret = gptlstart_gpu ('doalot_log')
    vals(n) = doalot (n)
    ret = gptlstop_gpu ('doalot_log')

    ret = gptlstart_gpu ('doalot_sqrt')
    vals(n) = doalot2 (n)
    ret = gptlstop_gpu ('doalot_sqrt')

    ret = gptlstart_gpu_c ('doalot_sqrt_c'//char(0))
    vals(n) = doalot2 (n)
    ret = gptlstop_gpu_c ('doalot_sqrt_c'//char(0))

    ret = gptlstart_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)
    vals(n) = doalot2 (n)
    ret = gptlstop_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)

    ret = gptlstart_handle_gpu_c ('a'//char(0), handle2)
    vals(n) = doalot2 (n)
    ret = gptlstop_handle_gpu_c ('a'//char(0), handle2)
  end do
!$acc end parallel
  ret = gptlstop ('doalot_cpu')

  ret = gptlstart ('doalot_cpu_nogputimers')
!$acc parallel loop copyout(vals)
  do n=1,1000
    vals(n) = doalot (n)
    vals(n) = doalot2 (n)
    vals(n) = doalot2 (n)
    vals(n) = doalot2 (n)
    vals(n) = doalot2 (n)
  end do
!$acc end parallel
  ret = gptlstop ('doalot_cpu_nogputimers')
  ret = gptlpr (0)
end program persist

real function doalot (n) result (sum)
  implicit none
  integer, intent(in) :: n
  integer :: i, iter
  real :: sum
!$acc routine seq

  sum = 0.
  do iter=1,10000
    do i=1,n
      sum = sum + log (real (iter*i))
    end do
  end do
end function doalot

real function doalot2 (n) result (sum)
  implicit none
  integer, intent(in) :: n
  integer :: i, iter
  real :: sum
!$acc routine seq

  sum = 0.
  do iter=1,10000
    do i=1,n
      sum = sum + sqrt (real (iter*i))
    end do
  end do
end function doalot2
