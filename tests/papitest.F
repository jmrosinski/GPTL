      program papitest
      implicit none

#include "../gpt.inc"
#ifdef HAVE_PAPI
#include <f77papi.h>
#endif

      integer ret, iter, niter
      integer papioption
      integer gptoption
      integer val
      logical done
      
      if (gptsetoption (gptcpu, 0) .lt. 0) stop 99
      if (gptsetoption (gptwall, 0) .lt. 0) stop 99
      if (gptsetoption (gptabort_on_error, 1) .lt. 0) stop 99

#ifdef HAVE_PAPI
      call gptpapiprinttable ()
#endif

      done = .false.
      do while (.not. done)
        write(6,*)'Enter PAPI option to be enabled'
        write(6,*)'or positive number when done'
        read (5,*)papioption
        if (papioption .lt. 0) then
          if (gptsetoption (papioption, 1) .lt. 0) then
            write(6,*)'gptsetoption failure'
          end if
        else
          done = .true.
        end if
      end do

      write(6,*)'GPTwall           = 1'
      write(6,*)'GPTcpu            = 2'
      write(6,*)'GPTabort_on_error = 3'
      write(6,*)'GPToverhead       = 4'

      done = .false.
      do while (.not. done)
        write(6,*)'Enter GPT option to set, then 0 (off) or 1 (on)'
        write(6,*)'   or negative numbers when done'
        read (5,*)gptoption, val
        if (gptoption > 0) then
          if (gptsetoption (gptoption, val) .lt. 0) then
            write(6,*)'gptsetoption failure'
          end if
        else
          done = .true.
        end if
      end do
      
      if (gptinitialize () .lt. 0) stop 99

      write(6,*)'Enter iteration count for threaded loop'
      read (5,*)niter

      if (gptstart ('total') .lt. 0) stop 99

!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,niter
        call sub (iter)
      end do

      if (gptstop ('total') .lt. 0) stop 99
      if (gptpr (0) .lt. 0) call exit(99)

      if (gptreset () .lt. 0) stop 99
      if (gptstart ('total') .lt. 0) stop 99

!$OMP PARALLEL DO PRIVATE (iter)

      do iter=1,niter
        call sub (iter)
      end do

      if (gptstop ('total') .lt. 0) stop 99
      if (gptpr (1) .lt. 0) call exit(99)
      if (gptfinalize () .lt. 0) stop 99

      stop
      end program
          
      subroutine sub (iter)
      implicit none

#include "../gpt.inc"

      integer iter

      character*15 name
      integer ret
      real sum1, sum2, x
      integer arr1(128*1000), arr2(128*1000)

      integer i

      write(name,'(i4.4,a8)') iter,'*2000_+*'

      sum1 = 0.
      sum2 = 0.

      ret = gptstart (name)
      do i=1,iter*1000
        x = i
        sum1 = sum1 + 2.1*x
        sum2 = sum2 + 3.2*x
      end do
      ret = gptstop (name)
      write(6,*)'sum1,sum2=',sum1,sum2

      write(name,'(i4.4,a7)') iter,'*2000_+'

      sum1 = 0.
      sum2 = 0.

      ret = gptstart (name)
      do i=1,iter*1000
        x = i
        sum1 = sum1 + x
        sum2 = sum2 + x
      end do
      ret = gptstop (name)
      write(6,*)'sum1,sum2=',sum1,sum2

      write(name,'(i4.4,a7)') iter,'*2000_*'

      sum1 = 1.
      sum2 = 1.

      ret = gptstart (name)
      do i=1,iter*1000
        sum1 = sum1*1.00001
        sum2 = sum2*1.00002
      end do
      ret = gptstop (name)
      write(6,*)'sum1,sum2=',sum1,sum2

      write(name,'(i4.4,a7)') iter,'*2000_/'

      sum1 = 1.
      sum2 = 1.

      ret = gptstart (name)
      do i=1,iter*1000
        sum1 = sum1 / 1.00001
        sum2 = sum2 / 1.00002
      end do
      ret = gptstop (name)
      write(6,*)'sum1,sum2=',sum1,sum2

      call badcache (1, arr1, arr2)
      call badcache (2, arr1, arr2)
      call badcache (4, arr1, arr2)
      call badcache (8, arr1, arr2)
      call badcache (16, arr1, arr2)
      call badcache (32, arr1, arr2)
      call badcache (64, arr1, arr2)
      call badcache (128, arr1, arr2)

! hangs on IBM
#ifndef AIX
      ret = gptstart ('sleep1')
      call sleep(1)
      ret = gptstop ('sleep1')
#endif

      return
      end subroutine sub

      subroutine badcache (stride, arrin, arrot)
      implicit none

#include "../gpt.inc"

      integer stride
      integer arrin(stride,1000), arrot(stride,1000)

      character*15 name
      integer i, ret

      write(name,'(a11,i3.3)') 'stormiss1K_',stride

      ret = gptstart (name)
      do i=1,1000
        arrot(1,i) = 7.
      end do
      ret = gptstop (name)

      write(name,'(a11,i3.3)') 'loadmiss1K_',stride

      ret = gptstart (name)
      do i=1,1000
        arrot(1,1) = arrin(1,i) 
      end do
      ret = gptstop (name)

      return
      end subroutine badcache

