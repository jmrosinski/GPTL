module gptl_acc
! GPTL module file for user code. This file contains an interface block for  parameter
! checking and binding to C routines wherever no character strings are being passed

  implicit none
  public

! Function prototypes/bindings

  interface
    integer function gptlinit_handle_gpu (name, handle)
      character(len=*) :: name
      integer :: handle
!$acc routine seq
    end function gptlinit_handle_gpu

    integer function gptlstart_gpu (handle) bind(C,name="GPTLstart_gpu")
      use iso_c_binding, only: c_int
      integer(c_int), intent(in), VALUE :: handle
!$acc routine seq
    end function gptlstart_gpu

    integer function gptlstop_gpu (handle) bind(C,name="GPTLstop_gpu")
      use iso_c_binding, only: c_int
      integer(c_int), intent(in), VALUE :: handle
!$acc routine seq
    end function gptlstop_gpu

    integer function gptlmy_sleep (seconds) bind(C,name="GPTLmy_sleep")
      use iso_c_binding, only: c_float
      real(c_float), intent(in), VALUE :: seconds
!$acc routine seq                                                               
    end function gptlmy_sleep

    subroutine gptldummy_gpu () bind(C,name="GPTLdummy_gpu")
!$acc routine seq
    end subroutine gptldummy_gpu

    integer function gptlget_wallclock_gpu (handle, accum, maxval, minval) &
         bind(C,name="GPTLget_wallclock_gpu")
      use iso_c_binding, only: c_int, c_double

      integer(c_int), intent(in), VALUE :: handle
      real(c_double) :: accum
      real(c_double) :: maxval
      real(c_double) :: minval
!$acc routine seq
    end function gptlget_wallclock_gpu
    
    integer function gptlget_warp_thread (warp, thread) bind(C,name="GPTLget_warp_thread")
      use iso_c_binding, only: c_int
      integer(c_int) :: warp, thread
!$acc routine seq                                                               
    end function gptlget_warp_thread

    integer function gptlsliced_up_how (txt)
      character(len=*) :: txt
!$acc routine seq                                                               
    end function gptlsliced_up_how

    integer function gptlget_sm_thiswarp (smarr) bind(C,name="GPTLget_sm_thiswarp")
      integer :: smarr(:)
!$acc routine seq                                                               
    end function gptlget_sm_thiswarp

    integer function gptlcuprofilerstart () bind(C,name="GPTLcuProfilerStart")
!$acc routine seq                                                               
    end function gptlcuprofilerstart

    integer function gptlcuprofilerstop () bind(C,name="GPTLcuProfilerStop")
!$acc routine seq                                                               
    end function gptlcuprofilerstop
  end interface
end module gptl_acc
