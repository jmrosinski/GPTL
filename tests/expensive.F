      program expensive
      character*8 name

      include '../gpt.inc'

      call gptinitialize
!      call gptsetoption (usrsys, .false.)
      call gptstart('total')
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
        call gptstart('loop_100')
        sum = 0.
        do i=1,100000
          sum = sum + log(float(i))
        end do
        write(6,*)'sum=',sum
        call gptstop('loop_100')
      end do
      call gptstop('loop_99 ')
      call gptstop('total')
      call gptpr(0)
      stop
      end
