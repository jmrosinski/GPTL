      program opscost

      integer n

      real*8 sum(10)
      real*8 x

#include "../gpt.inc"

      if (gptsetoption (gptcpu, 1) .lt. 0) stop 99
      if (gptsetoption (gptwall, 1) .lt. 0) stop 99

      if (gptinitialize () .lt. 0) stop 99

      do n=1,10
        sum(n) = n
      end do

      if (gptstart ('multiplication') .lt. 0) stop 99
      do n=1,100000
        x = 1.1 + 0.0001*n
        sum(1) = sum(1) * x
        sum(2) = sum(2) * x
        sum(3) = sum(3) * x
        sum(4) = sum(4) * x
        sum(5) = sum(5) * x
        sum(6) = sum(6) * x
        sum(7) = sum(7) * x
        sum(8) = sum(8) * x
        sum(9) = sum(9) * x
        sum(10) = sum(10) * x
      end do
      if (gptstop ('multiplication') .lt. 0) stop 99

      if (gptstart ('addition') .lt. 0) stop 99
      do n=1,100000
        x = n + 0.01
        sum(1) = sum(1) + x
        sum(2) = sum(2) + x
        sum(3) = sum(3) + x
        sum(4) = sum(4) + x
        sum(5) = sum(5) + x
        sum(6) = sum(6) + x
        sum(7) = sum(7) + x
        sum(8) = sum(8) + x
        sum(9) = sum(9) + x
        sum(10) = sum(10) + x
      end do
      if (gptstop ('addition') .lt. 0) stop 99

      if (gptstart ('division') .lt. 0) stop 99
      do n=1,100000
        x = 1.1 + 0.0001*n
        sum(1) = sum(1) / x
        sum(2) = sum(2) / x
        sum(3) = sum(3) / x
        sum(4) = sum(4) / x
        sum(5) = sum(5) / x
        sum(6) = sum(6) / x
        sum(7) = sum(7) / x
        sum(8) = sum(8) / x
        sum(9) = sum(9) / x
        sum(10) = sum(10) / x
      end do
      if (gptstop ('division') .lt. 0) stop 99

      do n=1,10
        sum(n) = 0.
      end do

      if (gptstart ('exponentiation') .lt. 0) stop 99
      do n=1,100000
        x = -1. + 0.000001*n
        sum(1) = exp(sum(1)*x)
        sum(2) = exp(sum(2)*x)
        sum(3) = exp(sum(3)*x)
        sum(4) = exp(sum(4)*x)
        sum(5) = exp(sum(5)*x)
        sum(6) = exp(sum(6)*x)
        sum(7) = exp(sum(7)*x)
        sum(8) = exp(sum(8)*x)
        sum(9) = exp(sum(9)*x)
        sum(10) = exp(sum(10)*x)
      end do
      if (gptstop ('exponentiation') .lt. 0) stop 99

      if (gptstart ('power') .lt. 0) stop 99
      do n=1,100000
        x = 1. + 0.000001*n
        sum(1) = sum(1)**x
        sum(2) = sum(2)**x
        sum(3) = sum(3)**x
        sum(4) = sum(4)**x
        sum(5) = sum(5)**x
        sum(6) = sum(6)**x
        sum(7) = sum(7)**x
        sum(8) = sum(8)**x
        sum(9) = sum(9)**x
        sum(10) = sum(10)**x
      end do
      if (gptstop ('power') .lt. 0) stop 99

      if (gptpr(0) .lt. 0) stop 99
      if (gptfinalize () .lt. 0) stop 99
      stop
      end program
