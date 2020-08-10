program driver_mpi
  use mpi
  use openacc
  use getval, only: getval_int
  
  implicit none

  integer :: maxwarps_gpu = 112
  integer :: outerlooplen
  integer :: innerlooplen = 100
  integer :: mostwork = 1000
  integer :: balfact = 1
  integer :: myrank, ierr         ! MPI stuff
  integer :: ngpus, devicenum     ! gpu stuff

  call mpi_init (ierr)
  write(6,*)'calling mpi_comm_rank'
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  call mpi_comm_rank (MPI_COMM_WORLD, myrank, ierr)
  write(6,*)'myrank=',myrank
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  if (myrank == 0) then
    write(6,*)'All ranks calling acc_init'
  end if
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  call acc_init (ACC_DEVICE_NVIDIA)
  if (myrank == 0) then
    write(6,*)'All ranks calling acc_get_num_devices'
  end if
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  ngpus = acc_get_num_devices (ACC_DEVICE_NVIDIA)
  if (ngpus .eq. 0) then
    print *,'No GPUs found on this system.  Exiting'
    call mpi_barrier (MPI_COMM_WORLD, ierr)
    call mpi_abort (MPI_COMM_WORLD,1,ierr)
  endif
  write(6,*)'myrank=',myrank,' ngpus=',ngpus
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  devicenum = mod(myrank,ngpus)
  write(6,*)'myrank=', myrank, ' calling acc_set_device_num(',devicenum,')'
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  call acc_set_device_num (devicenum,ACC_DEVICE_NVIDIA)
  print *,'myrank=',myrank,' ngpus = ',ngpus,' device = ',acc_get_device_num(ACC_DEVICE_NVIDIA)
  call mpi_barrier (MPI_COMM_WORLD, ierr)

  if (myrank == 0) then
    call getval_int (mostwork, 'mostwork')
    call getval_int (maxwarps_gpu, 'maxwarps_gpu')
    outerlooplen = maxwarps_gpu * 32
    call getval_int (outerlooplen, 'outerlooplen')
    write(6,*)'outerlooplen=',outerlooplen
    call getval_int (innerlooplen, 'innerlooplen')
    call getval_int (balfact, 'balfact: 0=LtoR 1=balanced 2=RtoL')
  end if
  call mpi_bcast (mostwork,       1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  call mpi_bcast (maxwarps_gpu, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  call mpi_bcast (outerlooplen,   1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  call mpi_bcast (innerlooplen,   1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  call mpi_bcast (balfact,        1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

  call persist (myrank, mostwork, maxwarps_gpu, outerlooplen, innerlooplen, balfact)
  call mpi_finalize (ierr)
end program driver_mpi
