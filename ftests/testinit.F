      program testinit
      implicit none

#include "../gptl.inc"

      integer ret

      write(6,*)'testinit: Test enabling PAPI from GPTL...'
      ret = gptl_papilibraryinit ()

      if (ret == 0) then
        write(6,*)'Success'
      else
        write(6,*)'Failure'
        stop 999
      end if
      end program testinit
