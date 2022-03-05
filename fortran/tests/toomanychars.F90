program toomanychars
  use gptl
  implicit none
  
  integer :: handle
  integer :: ret
  real*8 :: val

  ! ASSUME MAX_CHARS < 64
  character(len=*), parameter :: &
       str = "0123456701234567012345670123456701234567012345670123456701234567"
  ret = gptlinitialize ()
  ret = gptlstart (str)
  if (ret == 0) then
    write(6,*) 'Uexpected success when passing in too many chars, GPTLstart returned ', ret
    stop 1
  else
    write(6,*) 'As expected when passing in too many chars, GPTLstart returned ', ret
    stop 0
  end if
end program toomanychars
