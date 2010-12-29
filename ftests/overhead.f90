program overhead
      
  use gptl

  implicit none

  integer :: ret, iter, i
  real :: t1, t2
  integer(8) :: handle = 0

  ret = gptlsetutr (gptlgettimeofday)
  ret = gptlinitialize ()

  call cpu_time (t1)
  do i=1,10000000
    ret = gptlstart ('loop')
    ret = gptlstop ('loop')
  end do
  call cpu_time (t2)
  write(6,'(a,1p,e10.2)') 'Time (sec) for 10e6 start/stop GPTL pairs using gettimeofday:', t2 - t1

  if (.false.) then
  
  ret = gptlfinalize ()
  ret = gptlsetutr (gptlnanotime)
  ret = gptlinitialize ()

  call cpu_time (t1)
  do i=1,10000000
    ret = gptlstart ('loop')
    ret = gptlstop ('loop')
  end do
  call cpu_time (t2)
  write(6,'(a,1p,e10.2)') 'Time (sec) for 10e6 start/stop GPTL pairs using nanotime:', t2 - t1
  
  ret = gptlfinalize ()
  ret = gptlsetutr (gptlgettimeofday)
  ret = gptlsetoption (gptlwall, 0)
  ret = gptlinitialize ()

  call cpu_time (t1)
  do i=1,10000000
    ret = gptlstart ('loop')
    ret = gptlstop ('loop')
  end do
  call cpu_time (t2)
  write(6,'(a,1p,e10.2)') 'Time (sec) for 10e6 start/stop GPTL pairs with wall disabled:', t2 - t1
  
  end if

  ret = gptlfinalize ()
  ret = gptlsetutr (gptlgettimeofday)
  ret = gptlsetoption (gptlwall, 0)
  ret = gptlinitialize ()

  call cpu_time (t1)
  do i=1,10000000
    ret = gptlstart_handle ('loop', handle)
    ret = gptlstop_handle ('loop', handle)
  end do
  call cpu_time (t2)
  write(6,'(a,1p,e10.2)') 'Time (sec) for 10e6 start_handle/stop_handle GPTL pairs with wall disabled:', t2 - t1
  stop

  ret = gptlfinalize ()
  ret = gptlsetutr (gptlgettimeofday)
  ret = gptlsetoption (gptlwall, 0)
  ret = gptlinitialize ()
  ret = gptldisable ()

  call cpu_time (t1)
  do i=1,10000000
    ret = gptlstart_handle ('loop', handle)
    ret = gptlstop_handle ('loop', handle)
  end do
  call cpu_time (t2)
  write(6,'(a,1p,e10.2)') 'Time (sec) for 10e6 start_handle/stop_handle GPTL pairs with gptldisable:', t2 - t1
  
  call cpu_time (t1)
  do i=1,10000000
    call do_nothing1 ('string1', handle)
    call do_nothing2 ('string2', handle)
    call do_nothing1 ('string3', handle)
    call do_nothing2 ('string4', handle)
  end do
  call cpu_time (t2)
  write(6,'(a,1p,e10.2)') 'Time (sec) for 40e6 calls to do_nothing:', t2 - t1
  stop 0
end program overhead

subroutine do_nothing1 (string, handle)
  implicit none

  character(len=*), intent(in) :: string
  integer(8), intent(in) :: handle

  if (string(1:1) == 'x') then
    write(6,*)'Bad string value'
  end if
end subroutine do_nothing1

subroutine do_nothing2 (string, handle)
  implicit none

  character(len=*), intent(in) :: string
  integer(8), intent(in) :: handle

  if (string(1:1) == 'x') then
    write(6,*)'Bad string value'
  end if
end subroutine do_nothing2
