subroutine sleep1 (outerlooplen, oversub)
  use gptl
  use gptl_acc
  implicit none

  integer, intent(in) :: outerlooplen
  integer, intent(in) :: oversub

  integer :: ret
  integer :: n, nn
  integer :: chunksize, nchunks
  integer :: sleep
  integer, parameter :: veclen = 1  ! parallel iteration count per outermost kernel iterator

  real*8 :: maxval, maxsav(0:outerlooplen-1)
  real*8 :: minval, minsav(0:outerlooplen-1)
  real*8 :: accum

  chunksize = min (outerlooplen, gptlcompute_chunksize (oversub, veclen))
  nchunks = ( outerlooplen + (chunksize-1) ) / chunksize;
  write(6,100)outerlooplen, nchunks, chunksize
100 format('outerlooplen=',i6,' broken into ',i6,' kernels of chunksize=', i6)
  write(6,*)'inner vector length (decreases chunksize if > warpsize)=', veclen

  n = 0
  do nn=0,outerlooplen-1,chunksize
    write(6,*)'chunk=', n, ' totalwork=', min (chunksize, outerlooplen - nn)
    n = n + 1
  end do
  
!$acc parallel copyout (ret, sleep)
  ret = gptlinit_handle_gpu ('sleep1'//char(0), sleep)
!$acc end parallel

  write(6,*)'Sleeping 1 second on GPU...'
  maxsav(:) = 1.
  minsav(:) = 1.
!$acc data copy (maxsav,minsav)

  ret = gptlstart ('sleep1ongpu')
  do nn=0,outerlooplen-1,chunksize
!$acc parallel loop private(ret,accum,maxval,minval) copyin(nn,chunksize,sleep)
    do n=nn,min(outerlooplen-1,nn+chunksize-1)
!      ret = gptlstart_gpu ('total_gputime')
      ret = gptlstart_gpu (sleep)
      ret = gptlmy_sleep (1.)
      ret = gptlstop_gpu (sleep)
      maxval = 1.
      minval = 1.
      ret = gptlget_wallclock_gpu (sleep, accum, maxval, minval)
      if (maxval > 1.1) then
        maxsav(n) = maxval
      end if
      if (minval < 0.9) then
        minsav(n) = minval
      end if
!      ret = gptlstop_gpu ('total_gputime')
    end do
!$acc end parallel
  end do
!$acc end data

  ret = gptlstop ('sleep1ongpu')

  do n=0,outerlooplen-1
    if (maxsav(n) > 1.1) then
      write(6,*)'maxsav(',n,')=',maxsav(n)
    end if
    if (minsav(n) < 0.9) then
      write(6,*)'minsav(',n,')=',minsav(n)
    end if
  end do
end subroutine sleep1
