program testpapi
  use gptl

  implicit none

#include <f90papi.h>
  integer :: narg        ! number of cmd line args
  integer :: ret         ! return code
  integer :: i
  integer :: code        ! PAPI counter code
  real(8) :: sum, val
  character(len=PAPI_MAX_STR_LEN) :: eventname = "PAPI_TOT_CYC"
  character(len=PAPI_MAX_STR_LEN) :: eventsave
  character(len=2) :: tooshort
  
  write(6,*)'testpapi: Testing PAPI interface...'

  ! Parse arg list: Just take the first one
  narg = command_argument_count()
  if (narg > 0) then
    call get_command_argument (1, eventname)
  end if

  write(6,*)'Testing gptlevent_name_to_code for ', trim(eventname)
  if (gptlevent_name_to_code (trim(eventname), code) /= 0) then
    write(6,*)'Failure from gptlevent_name_to_code'
    call exit(1)
  end if
  write(6,*)'Success: code for ',trim(eventname),'=',code

  write(6,*)'Testing gptlsetoption(',code,',1)'
  if (gptlsetoption (code, 1) /= 0) then
    write(6,*)'Failure'
    call exit(1)
  end if
  
  write(6,*)'Testing duplicate enable ', trim(eventname)
  if (gptlsetoption (code, 1) == 0) then
    write(6,*)'Failure to fail!'
    call exit(1)
  end if
  write(6,*)'Succeeded at failing!'
  
  write(6,*)'Testing turning off an already-on counter...'
  if (gptlsetoption (code, 0) == 0) then
    write(6,*)'Failure'
    call exit(1)
  end if
  write(6,*)'Succeeded at failing!'
  
  write(6,*)'Testing gptlevent_code_to_name for ', trim(eventname)
  if (gptlevent_code_to_name (code, eventsave) /= 0) then
    write(6,*)'Failure from gptlevent_code_to_name'
    call exit(1)
  end if

  if (trim(eventsave) == trim(eventname)) then
    write(6,*)'Success'
  else
    write(6,*)'Failure: got ',trim(eventsave),' expected ',trim(eventname)
    call exit(1)
  end if
  
  write(6,*)'Testing too short var for gptlevent_code_to_name...'
  if (gptlevent_code_to_name (code, tooshort) == 0) then
    write(6,*)'Failure of gptlevent_code_to_name to fail'
    call exit(1)
  end if
  write(6,*)'Success at catching too short output var name'
  
  write(6,*)'Testing bogus input to gptlevent_name_to_code...'
  if (gptlevent_name_to_code ('zzz', code) == 0) then
    write(6,*)'Failure of gptlevent_name_to_code to fail'
    call exit(1)
  end if
  write(6,*)'Success at catching bogus input name'
  
  write(6,*)'Testing bogus input to gptlevent_code_to_name...'
  code = -1
  if (gptlevent_code_to_name (code, eventsave) == 0) then
    write(6,*)'Failure of gptlevent_code_to_name to fail'
    call exit(1)
  end if
  write(6,*)'Success at catching bogus input code'
  
  write(6,*)'Testing gptlinitialize'
  if (gptlinitialize () /= 0) then
    write(6,*)'Failure'
    call exit(1)
  end if
  
  write(6,*) 'testing GPTLstart'
  ret = gptlstart ('sum')
  if (ret /= 0) then
    write(6,*)'Unexpected failure from gptlstart'
    call exit(3)
  end if

  sum = 0.
  do i=1,1000000
    sum = sum + i
  end do

  write(6,*) 'testing GPTLstop'
  ret = gptlstop ('sum')
  if (ret /= 0) then
    write(6,*)'Unexpected failure from gptlstop'
    call exit(3)
  end if
  
  write(6,*)'Testing gptlget_eventvalue...'
  if (gptlget_eventvalue ('sum', trim(eventname), 0, val) /= 0) then
    write(6,*)'Failure'
    call exit(1)
  end if
  write(6,*)'Success: counter=', val

  if (trim(eventname) == "PAPI_TOT_CYC") then
    write(6,*) "testing reasonableness of PAPI counters..."
    if (val < 1 .or. val > 1.e9) then
      write(6,*)'Suspicious val=',val
      call exit(1)
    end if
  end if

  if (gptlpr (0) /= 0) then
    write(6,*) "bad return from GPTLpr(0)"
    stop 6;
  end if

  write(6,*) "All tests successful"
  stop 0
end program testpapi
