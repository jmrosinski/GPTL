module cubindings
  use iso_c_binding
  implicit  none

  interface

! start
    integer(c_int) function gptlstart_gpu_c(arg) bind(C,name="GPTLstart_gpu")
      character(kind=c_char,len=*) :: arg
    end function gptlstart_gpu_c

    integer(c_int) function gptlstart_handle_gpu_c(arg, handle) bind(C,name="GPTLstart_handle_gpu")
      character(kind=c_char,len=*), intent(in) :: arg
      integer(kind=c_int), intent(inout) :: handle
    end function gptlstart_handle_gpu_c

! stop  
    integer(c_int) function gptlstop_gpu_c(arg) bind(C,name="GPTLstop_gpu")
      character(kind=c_char,len=*), intent(in) :: arg
    end function gptlstop_gpu_c

    integer(c_int) function gptlstop_handle_gpu_c(arg, handle) bind(C,name="GPTLstop_handle_gpu")
      character(kind=c_char,len=*), intent(in) :: arg
      integer(kind=c_int), intent(inout) :: handle
    end function gptlstop_handle_gpu_c

  end interface
end module cubindings
