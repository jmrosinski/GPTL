      program cheap
      character*8 name

      call t_initializef
      do n=1,98
        if (n.lt.10) then
          write(name,'(a6,i1)') 'loop_0',n
        else
          write(name,'(a5,i2)') 'loop_',n
        end if
        call t_startf(name)
        call t_stopf(name)
      end do
      call t_startf('loop_99 ')
      do n=1,100
        call t_startf('loop_01 ')
        call t_stopf('loop_01 ')
      end do
      call t_stopf('loop_99 ')
      call t_prf(0)
      stop
      end
