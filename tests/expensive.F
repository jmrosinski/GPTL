      program expensive
      character*8 name

      include '../gpt.inc'

      if (gptsetoption (gptcpu, 1) < 0) stop 99
      if (gptsetoption (gptwall, 0) < 0) stop 99

      if (gptinitialize () < 0) stop 99

      if (gptstart('total') < 0) stop 99

      do n=1,98
        if (n.lt.10) then
          write(name,'(a6,i1)') 'loop_0',n
        else
          write(name,'(a5,i2)') 'loop_',n
        end if
        if (gptstart(name) < 0) stop 99
        if (gptstop(name) < 0) stop 99
      end do
      if (gptstart('loop_99 ') < 0) stop 99
      do n=1,100
        if (gptstart('loop_100') < 0) stop 99
        sum = 0.
        do i=1,100000
          sum = sum + log(float(i))
        end do
        write(6,*)'sum=',sum
        if (gptstop('loop_100') < 0) stop 99
      end do
      if (gptstop('loop_99 ') < 0) stop 99
      if (gptstop('total') < 0) stop 99
      if (gptpr(0) < 0) stop 99
      stop
      end
