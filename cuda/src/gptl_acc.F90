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
  end interface
end module gptl_acc
