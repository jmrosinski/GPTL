      program camtest
      implicit none

      include "../gptl.inc"
      double precision wall, usr, sys
      integer size, rss, share, text, datastack
      integer id
      integer ret

      ret = gptlsetoption (gptlwall, 0)
      ret = gptlsetoption (gptlcpu, 1)
      ret = gptlsetutr (gptlclockgettime)

      ret = gptlinitialize ()
      ret = gptlstamp (wall, usr, sys)
      write (6,*)'wall,usr,sys=', wall,usr,sys
      call t_startf ('total')
      ret =  gptldisable ()
      call t_startf ('sub1')
      call sub1 ()
      call t_stopf ('sub1')
      ret = gptlenable ()
      call t_stopf ('total')
      call t_prf (0)
      ret = gptlreset ()
      ret = gptlfinalize ()
      ret = gptlget_memusage (size, rss, share, text, datastack)
      write (6,*) 'size,rss,share,text,datastack='
      write (6,*) size,rss,share,text,datastack
      ret = gptlprint_memusage ('somewhere')
      ret = gptl_papiprinttable ()
      id = gptl_papiname2id ('PAPI_TOT_CYC')
      write(6,*)'id=',id

      stop 0
      end

      subroutine sub1
      implicit none
      
      integer i
      real x

      x = 0.
      do i=1,1000000
        x = x + sqrt (0.1*i)
      end do
      return
      end
