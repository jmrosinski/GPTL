      program expensive
      character*8 name

      include '../header.inc'

      call t_initializef
!      call t_setoptionf (usrsys, .false.)
      call t_startf('total')
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
        call t_startf('loop_100')
        sum = 0.
        do i=1,100000
          sum = sum + log(float(i))
        end do
        write(6,*)'sum=',sum
        call t_stopf('loop_100')
      end do
      call t_stopf('loop_99 ')
      call t_stopf('total')
      call t_prf(0)
      stop
      end
