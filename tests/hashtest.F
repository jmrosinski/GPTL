      program hashtest
      implicit none

#include "../gptl.inc"

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

      if (gptlsetoption (gptlabort_on_error, 1) .lt. 0) stop 99
      if (gptlinitialize () .lt. 0) stop 99

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

      if (gptlpr (0) .lt. 0) stop 77
      if (gptlfinalize () .lt. 0) stop 99
      stop
      end

      subroutine sub (start, ntpt)
      implicit none

#include "../gptl.inc"

      integer start
      integer ntpt
      character*9 name

      integer i

      do i=start,start+ntpt-1
        write (name,'(a5i4.4)') 'timer',i
        if (gptlstart (name) .lt. 0) stop 99
        if (gptlstop (name) .lt. 0) stop 99
      end do

      return
      end
