      program cheap
      character*8 name

      call gptinitialize ()
      do n=1,98
        if (n.lt.10) then
          write(name,'(a6,i1)') 'loop_0',n
        else
          write(name,'(a5,i2)') 'loop_',n
        end if
        call gptstart(name)
        call gptstop(name)
      end do
      call gptstart('loop_99 ')
      do n=1,100
        call gptstart('loop_01 ')
        call gptstop('loop_01 ')
      end do
      call gptstop('loop_99 ')
      call gptpr(0)
      stop
      end
