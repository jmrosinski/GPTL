      program hashtest
      implicit none

      include '../gpt.inc'

      integer niter
      integer ntpt
      integer start
      integer n
      integer ntimers

      write(6,*)'The reason for this code is to exercise the hash table'
      write(6,*)'Enter number of iterations for threaded loop:'
      read (5,*) niter
      write(6,*)'Enter total number of timers:'
      read (5,*) ntimers
      ntpt = ntimers / niter

      call gptsetoption (gptabort_on_error, 1)
      if (gptinitialize () < 0) stop 99

!$OMP PARALLEL DO PRIVATE (N, START)

      do n=1,niter
        start = (n-1)*ntpt
        call sub (start, ntpt)
      end do

!$OMP PARALLEL DO PRIVATE (N, START)

      do n=1,niter
        start = (n-1)*ntpt
        call sub (start, ntpt)
      end do

      if (gptpr (0) < 0) stop 77
      stop
      end

      subroutine sub (start, ntpt)
      integer start
      integer ntpt
      character*9 name

      integer i

      do i=start,start+ntpt-1
        write (name,'(a5i4.4)') 'timer',i
        call gptstart (name)
        call gptstop (name)
      end do

      return
      end
