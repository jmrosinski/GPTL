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

      if (gptsetoption (gptabort_on_error, 1) < 0) stop 99
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
      implicit none

      include '../gpt.inc'

      integer start
      integer ntpt
      character*9 name

      integer i

      do i=start,start+ntpt-1
        write (name,'(a5i4.4)') 'timer',i
        if (gptstart (name) < 0) stop 99
        if (gptstop (name) < 0) stop 99
      end do

      return
      end
