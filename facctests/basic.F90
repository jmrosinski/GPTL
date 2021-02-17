program basic
  use gptl
  use gptl_acc
  
  implicit none

  ! command argument parsing
  integer :: narg
  character(len=128) :: arg
  
  integer :: ret
  integer :: khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu
  real    :: oversub
  integer :: niter, n
  integer :: extraiter               ! number of extra iterations that don't complete a block
  integer :: nwarps
  integer :: warp, warpsav
  real    :: sleepsec = 1.           ! sleep time (defaut 1 sec)
  real*8  :: accummax, accummin      ! max/min times across warps
  real*8  :: wc                      ! wallclock measured on CPU
  integer :: total_gputime, sleep1   ! handles
  real*8, allocatable :: accum(:)
  ! GPU-local variables
  integer :: mywarp, mythread
  real*8  :: maxsav, minsav
  
  ret = gptlget_gpu_props (khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu)
  niter = cores_per_gpu

  narg = command_argument_count ()
  n = 1
  do while (n <= narg)
    call get_command_argument (n, arg)
    if (trim(arg) == '-n') then
      call get_command_argument (n+1, arg)
      read (arg,*) niter
      if (niter < 1) then
        write(6,*) 'niter must be > 0 ', niter, ' is invalid'
        stop -1
      end if
    else if (trim(arg) == '-s') then
      call get_command_argument (n+1, arg)
      read (arg,*) sleepsec
      if (sleepsec < 0.) then
        write(6,*) 'sleepsec cannot be < 0. ', sleepsec, ' is invalid'
        stop -1
      end if
    else
      write(6,*) 'unknown option ', arg
      call get_command_argument (0, arg)
      write(6,*) ('Usage: ', trim(arg), ' [-n niter] [-s sleepsec]')
      write(6,*) ('sleepsec can be fractional');
      stop 2;
    end if
    n = n + 2  ! All cmd-line args take an additional arg so increment n here
  end do

  oversub = real (niter) / cores_per_gpu
  write(6,*) 'oversubscription factor=', oversub

  nwarps = niter / warpsize
  if (mod (niter, warpsize) /= 0) then
    extraiter = warpsize - (niter - (nwarps*warpsize));
    nwarps = nwarps + 1
    write(6,*) 'Last iteration will be only ', extraiter, ' elements'
  end if

  ! Ensure that all warps will be examined by GPTL
  ret = gptlsetoption (gptlmaxwarps_gpu, nwarps)

  ! Use gettimeofday() as the underlying CPU timer. This is optional
  ret = gptlsetutr (gptlgettimeofday)

  ! Initialize the GPTL library on CPU and GPU
  ret = gptlinitialize ()

  allocate(accum(0:nwarps-1))
  accum(:) = 0

  ! Define handles
!$acc parallel private(ret) copyout(total_gputime,sleep1)
  ret = gptlinit_handle_gpu ('total_gputime'//char(0), total_gputime)
  ret = gptlinit_handle_gpu ('sleep1'//char(0),        sleep1)
!$acc end parallel
  ret = gptlcudadevsync ()

  write(6,*)'Sleeping ', sleepsec, ' seconds on GPU...'

  ret = gptlstart ('total')
!$acc parallel loop private(ret,mywarp,mythread,maxsav,minsav) &
!$acc&              copyin(total_gputime,sleep1,warpsize,sleepsec) copyout(accum)
  do n=0,niter-1
    ret = gptlsliced_up_how ('loop');
    ret = gptlstart_gpu (total_gputime)
    ret = gptlstart_gpu (sleep1)
    ret = gptlmy_sleep (sleepsec)
    ret = gptlstop_gpu (sleep1)
    ret = gptlget_warp_thread (mywarp, mythread)
    ret = gptlget_wallclock_gpu (sleep1, accum(mywarp), maxsav, minsav)
    ret = gptlstop_gpu (total_gputime)
  end do
!$acc end parallel
  
  ret = gptlcudadevsync ()
  ret = gptlstop ('total')

  ret = gptlget_wallclock ('total', -1, wc)
  write(6,'(a,f9.3)')'CPU says total wallclock=',wc,' seconds'

  accummax = 0.
  warpsav = -1
  do warp=0,nwarps-1
    if (accum(warp) > accummax) then
      accummax = accum(warp)
      warpsav = warp
    end if
  end do
  write(6,'(a,f12.9,a,i)') 'Max time slept=',accummax,' at warp=', warpsav

  accummin = 1.e36
  warpsav = -1
  do warp=0,nwarps-1
#ifdef DEBUG
    write(6,*)'accum(',warp,')=',accum(warp)
#endif
    if (accum(warp) < accummin) then
      accummin = accum(warp)
      warpsav = warp
    end if
  end do
  write(6,'(a,f12.9,a,i)') 'Min time slept=',accummin,' at warp=', warpsav

  ret = gptlpr (0)

  ret = gptlcudadevsync ();  ! Ensure printing of GPU results is complete before resetting
  ret = gptlreset ();        ! Reset CPU and GPU timers

  ret = gptlcudadevsync ();  ! Ensure resetting of timers is done before finalizing
  ret = gptlfinalize ();     ! Shutdown (incl. GPU)

  ret = gptlcudadevsync ();  ! Ensure any printing from GPTLfinalize_gpu is done before quitting
  stop 0
end program basic
