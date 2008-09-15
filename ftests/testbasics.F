      program testbasics
      implicit none

#include "../gptl.inc"

      integer :: ret
      integer :: nregions
      integer count, of       ! for gptlquery
      integer(8) :: pc        ! for gptlquery
      real(8) :: wc, usr, sys ! for gptlquery

      write(6,*)'testbasics: Testing basic GPTL usage...'
      write(6,*)'Testing gptlinitialize...'
      if (gptlinitialize () /= 0) then
        write(6,*)'Failure in gptlinitialize'
        stop 1
      end if
      write(6,*)'Success'
      write(6,*)'Testing gptlstart...'
      if (gptlstart ('testbasics') /= 0) then
        write(6,*)'Failure in gptlstart'
        stop 2
      end if
      write(6,*)'Success'
      write(6,*)'Testing gptlstop...'
      if (gptlstop ('testbasics') /= 0) then
        write(6,*)'Failure in gptlstop'
        stop 3
      end if
      write(6,*)'Success'
      write(6,*)'Testing gptlpr...'
      if (gptlpr (0) /= 0) then
        write(6,*)'Failure in gptlpr(0)'
        stop 4
      end if
      write(6,*)'Success'
      write(6,*)'Testing gptlprint_memusage...'
      if (gptlprint_memusage ('testbasics') /= 0) then
        write(6,*)'Failure in gptlprint_memusage'
        stop 5
      end if
      write(6,*)'Success'
      write(6,*)'Testing gptldisable/gptlenable...'
      if (gptldisable () /= 0) then
        write(6,*)'Failure in gptldisable'
        stop 6
      end if
      if (gptlstart ('zzz') /= 0) then
        write(6,*)'Failure in disabled gptlstart'
        stop 7
      end if
      if (gptlstop ('zzz') /= 0) then
        write(6,*)'Failure in disabled gptlstop'
        stop 8
      end if
      if (gptlenable () /= 0) then
        write(6,*)'Failure in gptlenable'
        stop 9
      end if
      write(6,*)'Sub-testing gptlget_nregions...'
      if (gptlget_nregions (0, nregions) /= 0) then
        write(6,*)'Failure in gptlget_nregions'
        stop 10
      end if
      if (nregions /= 1) then
        write(6,*)'Failure: expected nregions=1 got', nregions
        stop 11
      end if
      write (6,*)'Success'
      write (6,*)'Success'

      write(6,*)'Testing gptlquery...'
      ret = gptlquery ('testbasics', 0, count, of, wc, usr, sys, pc, 0)
      if (ret /= 0) then
        write(6,*)'Failure'
        stop 12
      end if
      if (count /= 1) then
        write(6,*)'Bad count value from gptlquery'
        stop 13
      end if
      if (of /= 0) then
        write(6,*)'Bad onflg value from gptlquery'
        stop 14
      end if
      write (6,*)'Success'

      write(6,*)'Testing gptlreset...'
      ret = gptlreset ()
      if (ret /= 0) then
        write(6,*)'Failure'
        stop 15
      end if
      ret = gptlquery ('testbasics', 0, count, of, wc, usr, sys, pc, 0)
      if (ret /= 0) then
        write(6,*)'Failure'
        stop 16
      end if
      if (count/=0 .or. of/=0 .or. wc/=0. .or. usr/=0.) then
        write(6,*)'Failure'
        stop 17
      end if
      write (6,*)'Success'

      write (6,*)'testbasics: All tests succeeded'
      end program testbasics
