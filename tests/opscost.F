      program opscost

      integer n

      real*8 sum(10)
      real*8 x

#include "../gptl.inc"

      if (gptlsetoption (gptlcpu, 1) .lt. 0) stop 99
      if (gptlsetoption (gptlwall, 1) .lt. 0) stop 99

      if (gptlinitialize () .lt. 0) stop 99

      do n=1,10
        sum(n) = n
      end do

      if (gptlstart ('multiplication') .lt. 0) stop 99
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
      if (gptlstop ('multiplication') .lt. 0) stop 99

      if (gptlstart ('addition') .lt. 0) stop 99
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
      if (gptlstop ('addition') .lt. 0) stop 99

      if (gptlstart ('division') .lt. 0) stop 99
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
      if (gptlstop ('division') .lt. 0) stop 99

      do n=1,10
        sum(n) = 0.
      end do

      if (gptlstart ('exponentiation') .lt. 0) stop 99
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
      if (gptlstop ('exponentiation') .lt. 0) stop 99

      if (gptlstart ('power') .lt. 0) stop 99
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
      if (gptlstop ('power') .lt. 0) stop 99

      if (gptlpr(0) .lt. 0) stop 99
      if (gptlfinalize () .lt. 0) stop 99
      stop
      end program
