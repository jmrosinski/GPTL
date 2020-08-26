program driver_serial
  use gptl
  use getval, only: getval_int
  implicit none

  integer :: maxwarps_gpu
  integer :: outerlooplen
  integer :: innerlooplen = 100
  integer :: mostwork = 1000
  integer :: balfact = 1
  integer :: oversub
  integer :: cores_per_sm
  integer :: cores_per_gpu
  integer :: khz, warpsize, devnum, smcount
  integer :: ret
  character(len=1) :: ans

  ret = gptlget_gpu_props (khz, warpsize, devnum, smcount, cores_per_sm, cores_per_gpu)
  write(6,*)'cores_per_gpu=', cores_per_gpu

  maxwarps_gpu = cores_per_gpu / warpsize
  call getval_int (maxwarps_gpu, 'maxwarps_gpu')
  write(6,*)'maxwarps_gpu=',maxwarps_gpu
  ret = gptlsetoption (gptlmaxwarps_gpu, maxwarps_gpu)

!  ret = gptlsetoption (gptlmaxtimers_gpu, 100)
!  ret = gptlsetoption (gptltablesize_gpu, 32)   ! This setting gives 1 collision
  write(6,*)'Calling gptlinitialize'
  ret = gptlinitialize ()

  call getval_int (mostwork, 'mostwork')

  outerlooplen = maxwarps_gpu * warpsize
  call getval_int (outerlooplen, 'outerlooplen')
  write(6,*)'outerlooplen=',outerlooplen

  call getval_int (innerlooplen, 'innerlooplen')
  call getval_int (balfact, 'balfact: 0=LtoR 1=balanced 2=RtoL')

  oversub = (outerlooplen + (cores_per_gpu-1)) / cores_per_gpu
  call getval_int (oversub, 'oversub: oversubsubscription factor')

  write(6,*)'Enter 1 to run just sleep, anything else to run the full "persist" suite'
  read(5,*) ans
  if (ans == '1') then
    call sleep1 (outerlooplen, oversub)
  else
    call persist (mostwork, outerlooplen, innerlooplen, balfact, oversub)
  end if
  ret = gptlpr (0)
end program driver_serial
