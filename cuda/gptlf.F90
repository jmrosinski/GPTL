module gptl_gpu
! GPTL module file for user code. Parameter values match their counterparts
! in gptl.h. This file also contains an interface block for parameter checking.
! Also: Some F90-only subroutines after the interface block

  implicit none
  public

! Function prototypes

  interface
!$acc routine seq
     integer function gptlstart_gpu (name)
       character(len=*) :: name
     end function gptlstart_gpu

!$acc routine seq
     integer function gptlinit_handle_gpu (name, handle)
       character(len=*) :: name
       integer :: handle
     end function gptlinit_handle_gpu

!$acc routine seq
     integer function gptlstart_handle_gpu (name, handle)
       character(len=*) :: name
       integer :: handle
     end function gptlstart_handle_gpu

!$acc routine seq
     integer function gptlstop_gpu (name)
       character(len=*) :: name
     end function gptlstop_gpu

!$acc routine seq
     integer function gptlstop_handle_gpu (name, handle)
       character(len=*) :: name
       integer :: handle
     end function gptlstop_handle_gpu
  end interface
end module gptl_gpu
