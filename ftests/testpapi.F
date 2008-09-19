      program testpapi
      implicit none

#include "../gptl.inc"
#include <f77papi.h>

      integer :: ret
      integer :: i, code
      integer(8) :: pc ! papi counters
      real(8) :: sum

      write(6,*)'testpapi: Testing PAPI interface...'
      ret = PAPI_VER_CURRENT
      call papif_library_init (ret)
      if (ret /= PAPI_VER_CURRENT) then
        write(6,*)'ret, PAPI_VER_CURRENT=', ret, PAPI_VER_CURRENT
        write(6,*)'Failure to initialize PAPI library...'
        stop 1
      end if
      
      call papif_event_name_to_code ('PAPI_TOT_CYC', code, ret)
      if (ret /= 0) then
        write(6,*)'Failure from papif_event_name_to_code'
        stop 2
      end if

      write(6,*)'Testing enabling PAPI_TOT_CYC...'
      if (gptlsetoption (code, 1) /= 0) then
        write(6,*)'Failure'
        stop 3
      end if
      write(6,*)'Success'

      write(6,*)'Testing gptlinitialize'
      if (gptlinitialize () /= 0) then
        write(6,*)'Failure'
        stop 3
      end if
      write(6,*)'Success'

      ret = gptlstart ('sum')
      sum = 0.
      do i=1,1000000
        sum = sum + i
      end do
      ret = gptlstop ('sum')
      write(6,*)'Testing gptlquerycounters...'
      if (gptlquerycounters ('sum', 0, pc) /= 0) then
        write(6,*)'Failure'
        stop 4
      end if

      write(6,*)'sum,pc=',sum, pc
      if (pc < 1 .or. pc > 1.e7) then
        write(6,*)'Suspicious pc value=',pc
        stop 5
      else
        write(6,*)'Success'
      end if

      end program testpapi
