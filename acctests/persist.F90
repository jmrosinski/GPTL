subroutine persist (myrank, mostwork, maxwarps_gpu, outerlooplen, innerlooplen, balfact, oversub)
  use gptl
  use gptl_acc
  implicit none
!$acc routine (doalot_log) seq
!$acc routine (doalot_log_inner) seq
!$acc routine (doalot_sqrt) seq
!$acc routine (doalot_sqrt_double) seq

  integer, intent(in) :: myrank
  integer, intent(in) :: mostwork
  integer, intent(in) :: maxwarps_gpu
  integer, intent(in) :: outerlooplen
  integer, intent(in) :: innerlooplen
  integer, intent(in) :: balfact
  integer, intent(in) :: oversub

  integer :: ret
  integer :: n, nn
  integer :: niter
  integer :: chunksize, nchunks
  integer, parameter :: veclen = 1  ! parallel iteration count per outermost kernel iterator

  integer :: handle, handle2
  real :: factor

  real :: vals(outerlooplen)
  real*8 :: dvals(outerlooplen)
  real, external :: doalot_log, doalot_log_inner, doalot_sqrt
  real*8, external :: doalot_sqrt_double

!JR NOTE: gptlinitialize call increases mallocable memory size on GPU. That call will fail
!JR if any GPU activity happens before the call to gptlinitialize
  ret = gptlsetoption (gptlmaxwarps_gpu, maxwarps_gpu)
!  ret = gptlsetoption (gptlmaxtimers_gpu, 100)
!  ret = gptlsetoption (gptltablesize_gpu, 32)   ! This setting gives 1 collision
  write(6,*)'persist: calling gptlinitialize'
  ret = gptlinitialize ()
  write(6,*)'Calling gptldummy_gpu: CUDA will barf if hashtable is no longer a valid pointer'
!$acc kernels
  call gptldummy_gpu (0)
!$acc end kernels
  
! GPU-specific init_handle routine needed because its tablesize likely differs from CPU
!$acc parallel copyout(ret,handle,handle2)
  ret = gptlinit_handle_gpu ('doalot_handle_sqrt_c', handle)
  ret = gptlinit_handle_gpu ('a', handle2)
!$acc end parallel

!$acc kernels
  call gptldummy_gpu (1)
!$acc end kernels

  chunksize = min (outerlooplen, gptlcompute_chunksize (oversub, veclen))
  nchunks = ( outerlooplen + (chunksize-1) ) / chunksize;
  write(6,100)outerlooplen, nchunks, chunksize
100 format('outerlooplen=',i6,' broken into ',i6,' kernels of chunksize=', i6)
  write(6,*)'inner vector length (decreases chunksize if > warpsize)=', veclen

  ret = gptlstart ('total_kerneltime')
  ret = gptlstart ('donothing')

  n = 0
  do nn=0,outerlooplen-1,chunksize
    write(6,*)'chunk=', n, ' totalwork=', min (chunksize, outerlooplen - nn)
    n = n + 1
  end do
  
  do nn=0,outerlooplen-1,chunksize
!$acc parallel loop copyin(nn,chunksize,balfact,handle,handle2) copyout(ret)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
      ret = gptlstart_gpu ('total_gputime')
      ret = gptlstart_gpu ('donothing')
      ret = gptlstop_gpu ('donothing')
      ret = gptlstop_gpu ('total_gputime')
    end do
!$acc end parallel
  end do
  ret = gptlstop ('donothing')
  ret = gptlstop ('total_kerneltime')

  ret = gptlstart ('total_kerneltime')
  ret = gptlstart ('doalot')
  do nn=0,outerlooplen-1,chunksize
!$acc parallel loop private(niter,factor) &
!$acc&  copyin(nn,chunksize,mostwork,innerlooplen,balfact,handle,handle2) &
!$acc&  copyout(ret, vals, dvals)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
      ret = gptlstart_gpu ('total_gputime')
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

      vals(n) = doalot_log_inner (niter, innerlooplen)

      ret = gptlstart_gpu ('doalot_sqrt')
      vals(n) = doalot_sqrt (niter, innerlooplen)
      ret = gptlstop_gpu ('doalot_sqrt')

      ret = gptlstart_gpu ('doalot_sqrt_double')
      dvals(n) = doalot_sqrt_double (niter, innerlooplen)
      ret = gptlstop_gpu ('doalot_sqrt_double')

      ret = gptlstart_gpu_c ('doalot_sqrt_c'//char(0))
      vals(n) = doalot_sqrt (niter, innerlooplen)
      ret = gptlstop_gpu_c ('doalot_sqrt_c'//char(0))

      ret = gptlstart_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)
      vals(n) = doalot_sqrt (niter, innerlooplen)
      ret = gptlstop_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)
      
      ret = gptlstart_handle_gpu_c ('a'//char(0), handle2)
      vals(n) = doalot_sqrt (niter, innerlooplen)
      ret = gptlstop_handle_gpu_c ('a'//char(0), handle2)
      ret = gptlstop_gpu ('total_gputime')
    end do
!$acc end parallel
  end do
  ret = gptlstop ('doalot')
  ret = gptlstop ('total_kerneltime')
  
  write(6,*)'Sleeping 1 second on GPU...'
  ret = gptlstart ('total_kerneltime')
  ret = gptlstart ('sleep1ongpu')
  do nn=0,outerlooplen-1,chunksize
!$acc parallel loop private(ret) copyin(nn,chunksize)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
      ret = gptlstart_gpu ('total_gputime')
      ret = gptlstart_gpu ('sleep1')
      ret = gptlmy_sleep (1.)
      ret = gptlstop_gpu ('sleep1')
      ret = gptlstop_gpu ('total_gputime')
    end do
!$acc end parallel
  end do

  ret = gptlstop ('sleep1ongpu')
  ret = gptlstop ('total_kerneltime')
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
    ret = gptlstart_gpu ('doalot_log_inner')
    do i=1,n
      sum = sum + log (real (iter*i))
    end do
    ret = gptlstop_gpu ('doalot_log_inner')
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
      sum = sum + sqrt (float (iter*i))
    end do
  end do
end function doalot_sqrt

real*8 function doalot_sqrt_double (n, innerlooplen) result (sum)
  implicit none
  integer, intent(in) :: n, innerlooplen
  integer :: i, iter
  real*8 :: sum
!$acc routine seq

  sum = 0.
  do iter=1,innerlooplen
    do i=1,n
      sum = sum + sqrt (dble (iter*i))
    end do
  end do
end function doalot_sqrt_double

