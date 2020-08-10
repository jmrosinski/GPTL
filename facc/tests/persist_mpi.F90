program persist
  use mpi
  use openacc
  use gptl
  use gptl_acc
  implicit none
!$acc routine (gptlinit_handle_gpu) seq
!$acc routine (doalot) seq
!$acc routine (doalot2) seq

  integer :: ret
  integer :: n
  integer :: maxthreads_gpu = 3584
  integer :: outerlooplen = 100000
  integer :: innerlooplen = 100
  integer :: ans
  integer :: handle, handle2
  integer :: myrank, ierr         ! MPI stuff
  integer :: ngpus, devicenum     ! gpu stuff
  real, allocatable :: vals(:)
  real, external :: doalot, doalot2

  call mpi_init (ierr)
  write(6,*)'calling mpi_comm_rank'
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  call mpi_comm_rank (MPI_COMM_WORLD, myrank, ierr)
  write(6,*)'myrank=',myrank
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  write(6,*)'calling acc_init'
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  call acc_init (ACC_DEVICE_NVIDIA)
  write(6,*)'calling acc_get_num_devices'
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
  write(6,*)'JR calling acc_set_device_num(',devicenum,')'
  call mpi_barrier (MPI_COMM_WORLD, ierr)
  call acc_set_device_num (devicenum,ACC_DEVICE_NVIDIA)
  print *,'ngpus = ',ngpus,' rank = ',myrank,' device = ',acc_get_device_num(ACC_DEVICE_NVIDIA)
  call mpi_barrier (MPI_COMM_WORLD, ierr)

  if (myrank == 0) then
    call getval (maxthreads_gpu, 'maxthreads_gpu')
    call getval (outerlooplen, 'outerlooplen')
    call getval (innerlooplen, 'innerlooplen')
  end if
  call mpi_bcast (maxthreads_gpu, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  call mpi_bcast (outerlooplen,   1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  call mpi_bcast (innerlooplen,   1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  allocate (vals(outerlooplen))

!JR NOTE: gptlinitialize call increases mallocable memory size on GPU. That call will fail
!JR if any GPU activity happens before the call to gptlinitialize
  ret = gptlsetoption (gptlmaxthreads_gpu, maxthreads_gpu)
  write(6,*)'persist_mpi myrank=',myrank,': calling gptlinitialize'
  ret = gptlinitialize ()
!JR Need to call GPU-specific init_handle routine because its tablesize may differ from CPU
!$acc kernels copyout(ret,handle,handle2)
  ret = gptlinit_handle_gpu ('doalot_handle_sqrt_c', handle)
  ret = gptlinit_handle_gpu ('a', handle2)
!$acc end kernels

  ret = gptlstart ('doalot_cpu')
!$acc parallel loop copyin(handle,handle2) copyout(ret, vals)
  do n=1,outerlooplen
    ret = gptlstart_gpu ('doalot_log')
    vals(n) = doalot (n, innerlooplen)
    ret = gptlstop_gpu ('doalot_log')

    ret = gptlstart_gpu ('doalot_sqrt')
    vals(n) = doalot2 (n, innerlooplen)
    ret = gptlstop_gpu ('doalot_sqrt')

    ret = gptlstart_gpu_c ('doalot_sqrt_c'//char(0))
    vals(n) = doalot2 (n, innerlooplen)
    ret = gptlstop_gpu_c ('doalot_sqrt_c'//char(0))

    ret = gptlstart_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)
    vals(n) = doalot2 (n, innerlooplen)
    ret = gptlstop_handle_gpu_c ('doalot_handle_sqrt_c'//char(0), handle)

    ret = gptlstart_handle_gpu_c ('a'//char(0), handle2)
    vals(n) = doalot2 (n, innerlooplen)
    ret = gptlstop_handle_gpu_c ('a'//char(0), handle2)
  end do
!$acc end parallel
  ret = gptlstop ('doalot_cpu')

  ret = gptlstart ('doalot_cpu_nogputimers')
!$acc parallel loop copyout(vals)
  do n=1,outerlooplen
    vals(n) = doalot (n, innerlooplen)
    vals(n) = doalot2 (n, innerlooplen)
    vals(n) = doalot2 (n, innerlooplen)
    vals(n) = doalot2 (n, innerlooplen)
    vals(n) = doalot2 (n, innerlooplen)
  end do
!$acc end parallel
  ret = gptlstop ('doalot_cpu_nogputimers')
  ret = gptlpr (myrank)
  call mpi_finalize (ierr)
end program persist

real function doalot (n, innerlooplen) result (sum)
  implicit none
  integer, intent(in) :: n, innerlooplen
  integer :: i, iter
  real :: sum
!$acc routine seq

  sum = 0.
  do iter=1,innerlooplen
    do i=1,n
      sum = sum + log (real (iter*i))
    end do
  end do
end function doalot

real function doalot2 (n, innerlooplen) result (sum)
  implicit none
  integer, intent(in) :: n, innerlooplen
  integer :: i, iter
  real :: sum
!$acc routine seq

  sum = 0.
  do iter=1,innerlooplen
    do i=1,n
      sum = sum + sqrt (real (iter*i))
    end do
  end do
end function doalot2

subroutine getval (arg, str)
  implicit none

  integer, intent(inout) :: arg
  character(len=*), intent(in) :: str

  integer :: ans

  write(6,*)'Enter ',str,' or -1 to accept default (',arg,')'
  read(5,*) ans
  if (ans /= -1) then
    arg = ans
  end if
  write(6,*) str,'=',arg
end subroutine getval
