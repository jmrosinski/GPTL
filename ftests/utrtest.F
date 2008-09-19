      program utrtest
      implicit none
      external sub
#ifdef HAVE_MPI
#include "mpif.h"
#endif
#include "../gptl.inc"
      double precision sum
      integer ret
      sum = 0.

#ifdef HAVE_MPI
      call mpi_init ()
#endif

      write(6,*) 'Purpose: estimate overhead of GPTL timing (UTR)'
      ret = gptlsetoption (gptlabort_on_error, 0)
      ret = gptlsetoption (gptlverbose, 0)

      ret = gptlsetutr (gptlmpiwtime)
      ret = gptlsetutr (gptlrtc)
      ret = gptlsetutr (gptlclockgettime)
      ret = gptlsetutr (gptlgettimeofday)
      ret = gptlsetutr (gptlpapitime)
      ret = gptlsetutr (gptlnanotime)
      
      ret = gptlinitialize ()

      ret = gptlstart ('total')
!      ret = GPTLdisable ()
      call sub (1, 10000000, "1x1e7", sum)
      call sub (10, 1000000, "10x1e6", sum)
      call sub (100, 100000, "100x1e5", sum)
      call sub (1000, 10000, "1000x1e4", sum)
      call sub (10000, 1000, "1e4x1000", sum)
      call sub (100000, 100, "1e5x100", sum)
      call sub (1000000, 10, "1e6x10", sum)
      call sub (10000000, 1, "1e7x1", sum)
!      ret = gptlenable ()
      ret = gptlstop ("total")

      ret = gptlpr (0)
      stop 0
      end program utrtest

      subroutine sub (outer, inner, name, sum)
      implicit none
#include "../gptl.inc"

      integer outer
      integer inner
      character(len=*) name
      double precision sum

      integer i, j, ret

      do i = 0, outer-1
        ret = gptlstart (name)
        do j = 0, inner-1
          sum = sum + j
        end do
        ret = gptlstop (name)
      end do

      return
      end
