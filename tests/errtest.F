      program errtest
      implicit none

#include "../gpt.inc"

      integer ret, iter
      integer val
      
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
        if (gptsetoption (7, 1) .lt. 0) write(6,*)'setoption failure'
        if (gptinitialize () .lt. 0) write(6,*)'initialize failure'
      else if (val .eq. 1) then
        if (gptsetoption (gptcpu, 7) .lt. 0) then
          write(6,*)'setoption failure'
        end if
        if (gptinitialize () .lt. 0) write(6,*)'initialize failure'
      else if (val .eq. 2) then
        if (gptinitialize () .lt. 0) write(6,*)'initialize failure'
        if (gptstart ('errtest') .lt. 0) write(6,*)'start failure'
        if (gptstart ('errtest') .lt. 0) write(6,*)'start failure'
      else if (val .eq. 3) then
        if (gptinitialize () .lt. 0) write(6,*)'initialize failure'
        if (gptstop ('errtest') .lt. 0) write(6,*)'stop failure'
      else if (val .eq. 4) then
        if (gptinitialize () .lt. 0) write(6,*)'initialize failure'
        if (gptstart ('errtest') .lt. 0) write(6,*)'start failure'
        if (gptstop ('errtest') .lt. 0) write(6,*)'stop failure'
        if (gptstop ('errtest') .lt. 0) write(6,*)'stop failure'
      else if (val .eq. 5) then
        if (gptstart ('errtest') .lt. 0) write(6,*)'start failure'
        if (gptstop ('errtest') .lt. 0) write(6,*)'stop failure'
        if (gptpr ('errtest') .lt. 0) write(6,*)'stop failure'
      end if

      stop 0
      end program errtest
