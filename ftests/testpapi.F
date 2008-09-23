      program testpapi
      implicit none

#include "../gptl.inc"

      integer :: ret
      integer :: i, code
      integer(8) :: pc ! papi counters
      real(8) :: sum

      write(6,*)'testpapi: Testing PAPI interface...'
      write(6,*)'Calling gptl_papilibraryinit...'
      ret = gptl_papilibraryinit ()
      if (ret /= 0) then
        write(6,*)'Failure from gptl_papilibraryinit...'
        call exit(1)
      end if
      write(6,*)'Success'
      
      write(6,*)'Calling papif_event_name_to_code...'
      call papif_event_name_to_code ('PAPI_TOT_CYC', code, ret)
      if (ret /= 0) then
        write(6,*)'Failure from papif_event_name_to_code'
        call exit(1)
      end if
      write(6,*)'Success'

      write(6,*)'Testing passing PAPI_TOT_CYC to gptlsetoption...'
      if (gptlsetoption (code, 1) /= 0) then
        write(6,*)'Failure'
        call exit(1)
      end if
      write(6,*)'Success'

      write(6,*)'Testing gptlinitialize'
      if (gptlinitialize () /= 0) then
        write(6,*)'Failure'
        call exit(1)
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
        call exit(1)
      end if

      write(6,*)'sum,pc=',sum, pc
      if (pc < 1 .or. pc > 1.e9) then
        write(6,*)'Suspicious pc value=',pc
        call exit(1)
      else
        write(6,*)'Success'
      end if

      end program testpapi
