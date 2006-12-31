      program errtest
      implicit none

#include "../gptl.inc"

      integer ret, iter
      integer val

      write(6,*)'Purpose: test error conditions'

 1    write(6,*)'Enter number for error test:'
      write(6,*)'0: bad option'
      write(6,*)'1: bad option value'
      write(6,*)'2: start while already started'
      write(6,*)'3: stop never started'
      write(6,*)'4: stop while already stopped'
      write(6,*)'5: instance not called'

      read (5,*) val
      if (val .lt. 0 .or. val .gt. 5) then
        write(6,*)'Val must be between 0 and 5'
        goto 1
      end if

      if (val .eq. 0) then
        if (gptlsetoption (7, 1) .lt. 0) write(6,*)'setoption failure'
        if (gptlinitialize () .lt. 0) write(6,*)'initialize failure'
      else if (val .eq. 1) then
        if (gptlsetoption (gptlcpu, 7) .lt. 0) then
          write(6,*)'setoption failure'
        end if
        if (gptlinitialize () .lt. 0) write(6,*)'initialize failure'
      else if (val .eq. 2) then
        if (gptlinitialize () .lt. 0) write(6,*)'initialize failure'
        if (gptlstart ('errtest') .lt. 0) write(6,*)'start failure'
        if (gptlstart ('errtest') .lt. 0) write(6,*)'start failure'
      else if (val .eq. 3) then
        if (gptlinitialize () .lt. 0) write(6,*)'initialize failure'
        if (gptlstop ('errtest') .lt. 0) write(6,*)'stop failure'
      else if (val .eq. 4) then
        if (gptlinitialize () .lt. 0) write(6,*)'initialize failure'
        if (gptlstart ('errtest') .lt. 0) write(6,*)'start failure'
        if (gptlstop ('errtest') .lt. 0) write(6,*)'stop failure'
        if (gptlstop ('errtest') .lt. 0) write(6,*)'stop failure'
      else if (val .eq. 5) then
        if (gptlstart ('errtest') .lt. 0) write(6,*)'start failure'
        if (gptlstop ('errtest') .lt. 0) write(6,*)'stop failure'
        if (gptlpr ('errtest') .lt. 0) write(6,*)'stop failure'
      end if

      if (gptlfinalize () .lt. 0) write(6,*)'gptlfinalize error'
      stop 0
      end program errtest
