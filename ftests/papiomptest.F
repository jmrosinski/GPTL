      program papioomptest
      implicit none

#include "../gptl.inc"

#ifdef HAVE_PAPI
#include <f77papi.h>
#endif

      real add
      real multiply
      real multadd
      real divide

      external add
      external multiply
      external multadd
      external divide

      integer nompiter
      integer looplen
      integer iter
      integer papiopt

      real ret
      real zero

      nompiter = 128
      looplen = 1000000

      write(6,*) 'Purpose: test known-length loops with various'
      write(6,*) 'floating point ops'
      write(6,*) 'Include PAPI and OpenMP, respectively, if enabled'

      write(6,*) 'Outer loop length (OMP)=', nompiter
      write(6,*) 'Inner loop length=', looplen

      if (gptlsetoption (gptlverbose, 0) .lt. 0) call exit (3)
      if (gptlsetoption (gptlabort_on_error, 1) .lt. 0) call exit(3)
#ifdef HAVE_PAPI
      if (gptlsetoption (PAPI_FP_OPS, 1) .lt. 0) call exit (4)
      if (gptlsetoption (PAPI_TOT_CYC, 1) .lt. 0) call exit (4)
#endif

      if (gptlsetoption (gptloverhead, 0) < 0) call exit (4)
      if (gptlsetoption (gptlnarrowprint, 1) < 0) call exit (4)
      
      if (gptlinitialize () .lt. 0) call exit(1)

      ret = gptlstart ('total')

!$OMP PARALLEL DO PRIVATE (iter, zero)

      do iter=1,nompiter
        ret = add (looplen, zero)
        ret = multiply (looplen, zero)
        ret = multadd (looplen, zero)
        ret = divide (looplen, zero)
      end do

      ret = gptlstop ('total')
      if (gptlpr (0) .lt. 0) call exit(2)
      if (gptlfinalize () .lt. 0) call exit(3)

      stop 0
      end

      real function add (looplen, zero)
#include "../gptl.inc"
      integer looplen
      real zero

      integer i
      character*128 string
      real xlooplen
      real val

      val = zero
      xlooplen = looplen
      write (string,'(1p,e10.3,a)') xlooplen, 'additions'

      if (gptlstart (string) .lt. 0) call exit(4)
      do i=1,looplen
        val = val + i
      end do
      if (gptlstop (string) .lt. 0) call exit(5)
      add = val
      return
      end

      real function multiply (looplen, zero)
#include "../gptl.inc"
      integer looplen
      real zero

      integer i
      character*128 string
      real xlooplen
      real val

      val = zero
      xlooplen = looplen
      write (string,'(1p,e10.3,a)') xlooplen, 'multiplies'

      if (gptlstart (string) .lt. 0) call exit(6)
      do i=1,looplen
        val = val * i
      end do
      if (gptlstop (string) .lt. 0) call exit(7)
      multiply = val
      return
      end

      real function multadd (looplen, zero)
#include "../gptl.inc"
      integer looplen
      real zero

      integer i
      character*128 string
      real xlooplen
      real val

      val = zero
      xlooplen = looplen
      write (string,'(1p,e10.3,a)') xlooplen, 'multadds'

      if (gptlstart (string) .lt. 0) call exit(8)
      do i=1,looplen
        val = val + i*1.1
      end do
      if (gptlstop (string) .lt. 0) call exit(9)
      multadd = val
      return
      end

      real function divide (looplen, zero)
#include "../gptl.inc"
      integer looplen
      real zero

      integer i
      character*128 string
      real xlooplen
      real val

      val = zero
      xlooplen = looplen
      write (string,'(1p,e10.3,a)') xlooplen, 'divides'

      if (gptlstart (string) .lt. 0) call exit(10)
      do i=1,looplen
        val = val / i
      end do
      if (gptlstop (string) .lt. 0) call exit(11)
      divide = val
      return
      end
