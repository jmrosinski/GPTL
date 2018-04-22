program driver_serial
  use getval, only: getval_int
  implicit none

  integer :: maxwarps_gpu = 112
  integer :: outerlooplen
  integer :: innerlooplen = 100
  integer :: mostwork = 1000
  integer :: balfact = 1
  integer :: oversub = 1

  call getval_int (mostwork, 'mostwork')
  call getval_int (maxwarps_gpu, 'maxwarps_gpu')
  outerlooplen = maxwarps_gpu * 32
  call getval_int (outerlooplen, 'outerlooplen')
  write(6,*)'outerlooplen=',outerlooplen
  call getval_int (innerlooplen, 'innerlooplen')
  call getval_int (balfact, 'balfact: 0=LtoR 1=balanced 2=RtoL')
  call getval_int (oversub, 'oversub: oversubsubscription factor')

  call persist (0, mostwork, maxwarps_gpu, outerlooplen, innerlooplen, balfact, oversub)
end program driver_serial
