program driver_serial
  use getval, only: getval_int
  implicit none

  integer :: maxthreads_gpu = 3584
  integer :: outerlooplen
  integer :: innerlooplen = 100
  integer :: mostwork = 1000
  integer :: balfact = 1

  call getval_int (mostwork, 'mostwork')
  call getval_int (maxthreads_gpu, 'maxthreads_gpu')
  outerlooplen = maxthreads_gpu
  call getval_int (outerlooplen, 'outerlooplen')
  write(6,*)'outerlooplen=',outerlooplen
  call getval_int (innerlooplen, 'innerlooplen')
  call getval_int (balfact, 'balfact: 0=LtoR 1=balanced 2=RtoL')

  call persist (0, mostwork, maxthreads_gpu, outerlooplen, innerlooplen, balfact)
end program driver_serial
