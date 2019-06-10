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

  integer :: total_gputime, donothing, doalot_log_handle, doalot_log_inner_handle, &
             doalot_sqrt_handle, doalot_sqrt_double_handle, sleep1
  real :: factor

  real :: vals(0:outerlooplen-1)
  real*8 :: dvals(0:outerlooplen-1)
  real*8 :: maxval, maxsav(0:outerlooplen-1)
  real*8 :: minval, minsav(0:outerlooplen-1)
  real*8 :: accum

  write(6,*)'Calling gptldummy_gpu'
!$acc kernels
  call gptldummy_gpu ()
!$acc end kernels

! Define handles
!$acc parallel private(ret) &
!$acc&  copyout(total_gputime,donothing,doalot_sqrt_handle,doalot_sqrt_double_handle, &
!$acc&          doalot_log_handle,doalot_log_inner_handle,sleep1)
  ret = gptlinit_handle_gpu ('total_gputime'//char(0),     total_gputime)
  ret = gptlinit_handle_gpu ('donothing'//char(0),         donothing)
  ret = gptlinit_handle_gpu ('doalot_sqrt'//char(0),       doalot_sqrt_handle)
  ret = gptlinit_handle_gpu ('doalot_sqrt_double'//char(0),doalot_sqrt_double_handle)
  ret = gptlinit_handle_gpu ('doalot_log'//char(0),        doalot_log_handle)
  ret = gptlinit_handle_gpu ('doalot_log_inner'//char(0),  doalot_log_inner_handle)
  ret = gptlinit_handle_gpu ('sleep1'//char(0),            sleep1)
!$acc end parallel

  write(6,*)'doalot_log_handle=',doalot_log_handle
  
!$acc kernels
  call gptldummy_gpu ()
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
!$acc parallel loop copyin(nn,chunksize,balfact,total_gputime,donothing) copyout(ret)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
      ret = gptlstart_gpu (total_gputime)
      ret = gptlstart_gpu (donothing)
      ret = gptlstop_gpu (donothing)
      ret = gptlstop_gpu (total_gputime)
    end do
!$acc end parallel
    ret = gptlcudadevsync ();
  end do

  ret = gptlstop ('donothing')
  ret = gptlstop ('total_kerneltime')

  ret = gptlstart ('total_kerneltime')
  ret = gptlstart ('doalot')

  write(6,*)'doalot_log_handle=',doalot_log_handle
  do nn=0,outerlooplen-1,chunksize
!$acc parallel loop private(niter,factor,ret) &
!$acc&  copyin(nn,chunksize,mostwork,innerlooplen,balfact,total_gputime,doalot_log_handle, &
!$acc&         doalot_log_inner_handle,doalot_sqrt_handle,doalot_sqrt_double_handle) &
!$acc&  copyout(vals, dvals)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
      ret = gptlstart_gpu (total_gputime)
      factor = real(n) / real(outerlooplen-1)
      select case (balfact)
      case (0)
        niter = int(factor * mostwork)
      case (1)
        niter = mostwork
      case (2)
        niter = mostwork - int(factor * mostwork)
      end select

      ret = gptlstart_gpu (doalot_log_handle)
      vals(n) = doalot_log (niter, innerlooplen)
      ret = gptlstop_gpu (doalot_log_handle)

      vals(n) = doalot_log_inner (niter, innerlooplen, doalot_log_inner_handle)

      ret = gptlstart_gpu (doalot_sqrt_handle)
      vals(n) = doalot_sqrt (niter, innerlooplen)
      ret = gptlstop_gpu (doalot_sqrt_handle)

      ret = gptlstart_gpu (doalot_sqrt_double_handle)
      dvals(n) = doalot_sqrt_double (niter, innerlooplen)
      ret = gptlstop_gpu (doalot_sqrt_double_handle)

      ret = gptlstop_gpu (total_gputime)
    end do
!$acc end parallel
    ret = gptlcudadevsync ();
  end do
  ret = gptlstop ('doalot')
  ret = gptlstop ('total_kerneltime')
  
  write(6,*)'Sleeping 1 second on GPU...'
  maxsav(:) = 1.
  minsav(:) = 1.
!$acc data copy (maxsav,minsav)

  ret = gptlstart ('total_kerneltime')
  ret = gptlstart ('sleep1ongpu')
  do nn=0,outerlooplen-1,chunksize
!$acc parallel loop private(ret,accum,maxval,minval) copyin(nn,chunksize,total_gputime,sleep1)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
      ret = gptlstart_gpu (total_gputime)
      ret = gptlstart_gpu (sleep1)
      ret = gptlmy_sleep (1.)
      ret = gptlstop_gpu (sleep1)
      maxval = 1.
      minval = 1.
      ret = gptlget_wallclock_gpu (sleep1, accum, maxval, minval)
      if (maxval > 1.1) then
        maxsav(n) = maxval
      end if
      if (minval < 0.9) then
        minsav(n) = minval
      end if
      ret = gptlstop_gpu (total_gputime)
    end do
!$acc end parallel
    ret = gptlcudadevsync ();
  end do
!$acc end data

  ret = gptlstop ('sleep1ongpu')
  ret = gptlstop ('total_kerneltime')

  do n=0,outerlooplen-1
    if (maxsav(n) > 1.1) then
      write(6,*)'maxsav(',n,')=',maxsav(n)
    end if
    if (minsav(n) < 0.9) then
      write(6,*)'minsav(',n,')=',minsav(n)
    end if
  end do
end subroutine persist

