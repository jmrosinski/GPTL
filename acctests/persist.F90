subroutine persist (mostwork, outerlooplen, innerlooplen, balfact, oversub)
  use gptl
  use gptl_acc
  use subs
  
  implicit none

  integer, intent(in) :: mostwork
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

  real :: vals(0:outerlooplen-1)
  real*8 :: dvals(0:outerlooplen-1)
  real*8 :: maxval, maxsav(0:outerlooplen-1)
  real*8 :: minval, minsav(0:outerlooplen-1)
  real*8 :: accum

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

  ret = gptlstart ('total_kerneltime'//char(0))
  ret = gptlstart ('donothing'//char(0))

  n = 0
  do nn=0,outerlooplen-1,chunksize
    write(6,*)'chunk=', n, ' totalwork=', min (chunksize, outerlooplen - nn)
    n = n + 1
  end do
  
  do nn=0,outerlooplen-1,chunksize
!$acc parallel loop copyin(nn,chunksize,balfact,handle,handle2) copyout(ret)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
      ret = gptlstart_gpu ('total_gputime'//char(0))
      ret = gptlstart_gpu ('donothing'//char(0))
      ret = gptlstop_gpu ('donothing'//char(0))
      ret = gptlstop_gpu ('total_gputime'//char(0))
    end do
!$acc end parallel
    ret = gptlcudadevsync ();
  end do
  ret = gptlstop ('donothing'//char(0))
  ret = gptlstop ('total_kerneltime'//char(0))

  ret = gptlstart ('total_kerneltime'//char(0))
  ret = gptlstart ('doalot'//char(0))
  do nn=0,outerlooplen-1,chunksize
!$acc parallel loop private(niter,factor) &
!$acc&  copyin(nn,chunksize,mostwork,innerlooplen,balfact,handle,handle2) &
!$acc&  copyout(ret, vals, dvals)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
      ret = gptlstart_gpu ('total_gputime'//char(0))
      factor = real(n) / real(outerlooplen-1)
      select case (balfact)
      case (0)
        niter = int(factor * mostwork)
      case (1)
        niter = mostwork
      case (2)
        niter = mostwork - int(factor * mostwork)
      end select

      ret = gptlstart_gpu ('doalot_log'//char(0))
      vals(n) = doalot_log (niter, innerlooplen)
      ret = gptlstop_gpu ('doalot_log'//char(0))

      vals(n) = doalot_log_inner (niter, innerlooplen)

      ret = gptlstart_gpu ('doalot_sqrt'//char(0))
      vals(n) = doalot_sqrt (niter, innerlooplen)
      ret = gptlstop_gpu ('doalot_sqrt'//char(0))

      ret = gptlstart_gpu ('doalot_sqrt_double'//char(0))
      dvals(n) = doalot_sqrt_double (niter, innerlooplen)
      ret = gptlstop_gpu ('doalot_sqrt_double'//char(0))

      ret = gptlstart_gpu ('doalot_sqrt_c'//char(0))
      vals(n) = doalot_sqrt (niter, innerlooplen)
      ret = gptlstop_gpu ('doalot_sqrt_c'//char(0))

      ret = gptlstart_handle_gpu ('doalot_handle_sqrt_c'//char(0), handle)
      vals(n) = doalot_sqrt (niter, innerlooplen)
      ret = gptlstop_handle_gpu ('doalot_handle_sqrt_c'//char(0), handle)
      
      ret = gptlstart_handle_gpu ('a'//char(0), handle2)
      vals(n) = doalot_sqrt (niter, innerlooplen)
      ret = gptlstop_handle_gpu ('a'//char(0), handle2)
      ret = gptlstop_gpu ('total_gputime')
    end do
!$acc end parallel
    ret = gptlcudadevsync ();
  end do
  ret = gptlstop ('doalot'//char(0))
  ret = gptlstop ('total_kerneltime'//char(0))
  
  write(6,*)'Sleeping 1 second on GPU...'
  maxsav(:) = 1.
  minsav(:) = 1.
!$acc data copy (maxsav,minsav)

  ret = gptlstart ('total_kerneltime'//char(0))
  ret = gptlstart ('sleep1ongpu'//char(0))
  do nn=0,outerlooplen-1,chunksize
!$acc parallel loop private(ret,accum,maxval,minval) copyin(nn,chunksize)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
      ret = gptlstart_gpu ('total_gputime'//char(0))
      ret = gptlstart_gpu ('sleep1'//char(0))
      ret = gptlmy_sleep (1.)
      ret = gptlstop_gpu ('sleep1')
      maxval = 1.
      minval = 1.
      ret = gptlget_wallclock_gpu ('sleep1'//char(0), accum, maxval, minval)
      if (maxval > 1.1) then
        maxsav(n) = maxval
      end if
      if (minval < 0.9) then
        minsav(n) = minval
      end if
      ret = gptlstop_gpu ('total_gputime'//char(0))
    end do
!$acc end parallel
    ret = gptlcudadevsync ();
  end do
!$acc end data

  ret = gptlstop ('sleep1ongpu'//char(0))
  ret = gptlstop ('total_kerneltime'//char(0))

  do n=0,outerlooplen-1
    if (maxsav(n) > 1.1) then
      write(6,*)'maxsav(',n,')=',maxsav(n)
    end if
    if (minsav(n) < 0.9) then
      write(6,*)'minsav(',n,')=',minsav(n)
    end if
  end do
end subroutine persist

