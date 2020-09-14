program errtest
  use gptl

  implicit none

  write(6,*)'Purpose: test error conditions'

  ! Turn off verbose.
  if (gptlsetoption (GPTLverbose, 0) < 0) stop 2
  
  write(6,*)'testing bad option...'
  if (gptlsetoption (100, 1) == 0) stop 2
  write(6,*) 'ok!'

  if (gptlinitialize () < 0) stop 5
  write(6,*)'testing stop never started...'
  if (gptlstop ('errtest') == 0) stop 6
  write(6,*) 'ok!'

  write(6,*)'testing stop while already stopped...'
  if (gptlstart ('errtest') < 0) stop 9
  if (gptlstop ('errtest') < 0) stop 10
  if (gptlstop ('errtest') == 0) stop 11
  write(6,*) 'ok!'

  stop 0
end program errtest

