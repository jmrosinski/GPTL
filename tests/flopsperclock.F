      program flopsperclock
      implicit none

#include "../gptl.inc"

      integer iter, ret, n

      real*8 wall1, wall2, usr1, sys1, usr2, sys2, t1, t2

      real*8 sum0, sum1, overhead
      real*8 t1actual, t2actual

      real*8 x1,x2,x3,x4,x5,x6,x7,x8,x9,x10
      real*8 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10
      real*8 z1,z2,z3,z4,z5,z6,z7,z8,z9,z10

      if (gptlsetoption (gptlcpu, 0) < 0) stop 99

      if (gptlinitialize () < 0) stop 99

      open (unit=7, file='junk', form='formatted')

      call setvals (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,
     $              y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,
     $              z1,z2,z3,z4,z5,z6,z7,z8,z9,z10)

      do iter=1,10

        if (gptlstamp (wall1, usr1, sys1) < 0) stop 99
        do n=1,1000000
          call sub
        end do
        if (gptlstamp (wall2, usr2, sys2) < 0) stop 99

        overhead = wall2 - wall1

        sum0 = 0.

        if (gptlstamp (wall1, usr1, sys1) < 0) stop 99
        do n=1,1000000
          call sub

          sum0 = sum0 + x1 
          sum0 = sum0 + x2
          sum0 = sum0 + x3
          sum0 = sum0 + x4
          sum0 = sum0 + x5
          sum0 = sum0 + x6
          sum0 = sum0 + x7
          sum0 = sum0 + x8
          sum0 = sum0 + x9
          sum0 = sum0 + x10
        end do
        if (gptlstamp (wall2, usr2, sys2) < 0) stop 99
        t1 = wall2 - wall1

        write(7,*)'sum=',sum0

        sum1 = 0.

        if (gptlstamp (wall1, usr1, sys1) < 0) stop 99
        do n=1,1000000
          call sub

          sum0 = sum0 + 1.1*x1 
          sum1 = sum1 + 1.1*y1 
          sum0 = sum0 + 1.1*x2
          sum1 = sum1 + 1.1*y2
          sum0 = sum0 + 1.1*x3
          sum1 = sum1 + 1.1*y3
          sum0 = sum0 + 1.1*x4
          sum1 = sum1 + 1.1*y4
          sum0 = sum0 + 1.1*x5
          sum1 = sum1 + 1.1*y5
          sum0 = sum0 + 1.1*x6
          sum1 = sum1 + 1.1*y6
          sum0 = sum0 + 1.1*x7
          sum1 = sum1 + 1.1*y7
          sum0 = sum0 + 1.1*x8
          sum1 = sum1 + 1.1*y8
          sum0 = sum0 + 1.1*x9
          sum1 = sum1 + 1.1*y9
          sum0 = sum0 + 1.1*x10
          sum1 = sum1 + 1.1*y10
        end do
        if (gptlstamp (wall2, usr2, sys2) < 0) stop 99
        t2 = wall2 - wall1

        write(7,*)'sum=',sum0,sum1

        if (overhead > 0.8*t1 .or. overhead > 0.8*t2) then
          write(6,*)'Overhead is enormous'
!          cycle
        end if

        t1actual = t1 - overhead
        t2actual = t2 - overhead
        
        write(6,*)'iter=',iter,' est. flops/clock=',4.*t1actual/t2actual
        write(6,*)'Clock rate est=',10000000/t1actual*1.e-6,' MHz'
        write(6,*)'overhead =',overhead/t1*100.,'% of short loop and ',
     $                         overhead/t2*100.,'% of long loop'
      end do

      if (gptlfinalize () .lt. 0) stop 99
      stop
      end

      subroutine sub
      return
      end

      subroutine setvals(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,
     $                   y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,
     $                   z1,z2,z3,z4,z5,z6,z7,z8,z9,z10)

      real*8 x1,x2,x3,x4,x5,x6,x7,x8,x9,x10
      real*8 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10
      real*8 z1,z2,z3,z4,z5,z6,z7,z8,z9,z10

      x1 = 3.2
      x2 = 3.2
      x3 = 3.2
      x4 = 3.2
      x5 = 3.2
      x6 = 3.2
      x7 = 3.2
      x8 = 3.2
      x9 = 3.2
      x10 = 3.2

      y1 = 3.2
      y2 = 3.2
      y3 = 3.2
      y4 = 3.2
      y5 = 3.2
      y6 = 3.2
      y7 = 3.2
      y8 = 3.2
      y9 = 3.2
      y10 = 3.2

      z1 = 3.2
      z2 = 3.2
      z3 = 3.2
      z4 = 3.2
      z5 = 3.2
      z6 = 3.2
      z7 = 3.2
      z8 = 3.2
      z9 = 3.2
      z10 = 3.2

      return
      end


