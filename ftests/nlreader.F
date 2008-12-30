      program nlreader
      implicit none

#include "../gptl.inc"

      integer :: ret

      write(6,*)'nlreader: Testing gptlprocess_namelist...'
      call gptlprocess_namelist ('gptlnl', 1, ret)
      if (ret /= 0) then
        write(6,*)'Failure'
        call exit (1)
      end if
      
      ! Now turn off verbosity

      ret = gptlsetoption (gptlverbose, 0)
      ret = gptlinitialize ()
      write(6,*)'Success'
      end program nlreader
