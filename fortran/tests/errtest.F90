program errtest
  use gptl

  implicit none

  write(6,*)'Purpose: test error conditions'

  ! Turn off verbose.
  if (gptlsetoption (GPTLverbose, 0) < 0) stop 2
  
  write(6,*)'testing bad option...'
  if (gptlsetoption (100, 1) .eq. 0) stop 2
  if (gptlinitialize () < 0) stop 3
  if (gptlfinalize () < 0) stop 4
  write(6,*) 'ok!'

  write(6,*)'testing stop never started...'
  if (gptlinitialize () < 0) stop 5
  if (gptlstop ('errtest') .eq. 0) stop 6
  if (gptlfinalize () < 0) stop 7
  write(6,*) 'ok!'

  write(6,*)'testing stop while already stopped...'
  if (gptlinitialize () < 0) stop 8
  if (gptlstart ('errtest') < 0) stop 9
  if (gptlstop ('errtest') < 0) stop 10
  if (gptlstop ('errtest') .eq. 0) stop 11
  write(6,*) 'ok!'

  write(6,*)'testing instance not called...'
  if (gptlstart ('errtest') < 0) stop 12
  if (gptlstop ('errtest') < 0) stop 13
  if (gptlpr (0) < 0) stop 14
  write(6,*) 'ok!'

  stop 0
end program errtest

