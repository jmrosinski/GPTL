module myvars
  integer :: iam
end module myvars

program pmpi
  use myvars
  implicit none

#include <mpif.h>
#include "../gptl.inc"

  integer, parameter :: tag = 98
  integer, parameter :: count = 1024

  integer :: i, j, ret
  integer :: commsize
  integer :: val
  integer :: comm = MPI_COMM_WORLD
  integer :: sendbuf(0:count-1)
  integer :: recvbuf(0:count-1)
  integer :: sum
  integer :: status(MPI_STATUS_SIZE)
  integer :: request
  integer :: dest
  integer :: source
  integer :: rdispls(0:count-1)
  integer :: sdispls(0:count-1)

  integer, allocatable :: atoabufsend(:)
  integer, allocatable :: atoabufrecv(:)
  integer, allocatable :: gsbuf(:,:)      ! gather/scatter buffer
  integer, allocatable :: recvcounts(:)
  integer, allocatable :: sendcounts(:)

  logical :: flag
  integer :: debugflag = 1

  ret = gptlsetoption (gptloverhead, 0)
  ret = gptlsetoption (gptlpercent, 0)
  ret = gptlsetoption (gptlabort_on_error, 1)
  ret = gptlsetoption (gptlsync_mpi, 1)

#if ( ! defined HAVE_IARGCGETARG )
  ret = gptlinitialize ()
  ret = gptlstart ("total")
#endif

  call mpi_init (ret)

! For debugging, go into infinite loop so debugger can attach and reset
#ifdef DEBUG
  do while (debugflag == 1)
  end do
#endif
  call mpi_comm_rank (comm, iam, ret)
  call mpi_comm_size (comm, commsize, ret)
  
  do i=0,count-1
    sendbuf(i) = iam
  end do

  dest = mod ((iam + 1), commsize)
  source = iam - 1
  if (source < 0) then
    source = commsize - 1
  end if
!
! mpi_send
! mpi_recv
! mpi_probe
!
  if (mod (commsize, 2) == 0) then
    if (iam == 0) then
      write(6,*)'pmpi.F90: testing send, recv, probe...'
    end if

    if (mod (iam, 2) == 0) then
      write(6,*)'even calling send'
      call mpi_send (sendbuf, count, MPI_INTEGER, dest, tag, comm, ret)
      write(6,*)'even calling recv'
      call mpi_recv (recvbuf, count, MPI_INTEGER, source, tag, comm, status, ret)
    else
      write(6,*)'odd calling probe'
      call mpi_probe (source, tag, comm, status, ret)
      if (ret /= MPI_SUCCESS) then
        write(6,*) "iam=", iam, " mpi_probe: bad status return"
        call mpi_abort (MPI_COMM_WORLD, -1, ret)
      end if
      write(6,*)'odd calling recv'
      call mpi_recv (recvbuf, count, MPI_INTEGER, source, tag, comm, status, ret)
      write(6,*)'odd calling send'
      call mpi_send (sendbuf, count, MPI_INTEGER, dest, tag, comm, ret)
    end if
    call chkbuf ('mpi_send + mpi_recv', recvbuf(:), count, source)

    if (iam == 0) then
      write(6,*)'Success'
      write(6,*)'pmpi.F90: testing ssend...'
    end if
    stop 0
!
! mpi_ssend
!
    if (mod (iam, 2) == 0) then
      call mpi_ssend (sendbuf, count, MPI_INTEGER, dest, tag, comm, ret)
      call mpi_recv (recvbuf, count, MPI_INTEGER, source, tag, comm, status, ret)
    else
      call mpi_recv (recvbuf, count, MPI_INTEGER, source, tag, comm, status, ret)
      call mpi_ssend (sendbuf, count, MPI_INTEGER, dest, tag, comm, ret)
    end if
    call chkbuf ('mpi_send + mpi_recv', recvbuf(:), count, source)
    if (iam == 0) then
      write(6,*)'Success'
      write(6,*)'pmpi.F90: testing sendrecv...'
    end if
  end if
!
! mpi_sendrecv
! 
  call mpi_sendrecv (sendbuf, count, MPI_INTEGER, dest, tag, &
                     recvbuf, count, MPI_INTEGER, source, tag, &
                     comm, status, ret)
  call chkbuf ('mpi_sendrecv', recvbuf(:), count, source)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing irecv, isend, iprobe, itest, wait, waitall...'
  end if
!
! mpi_irecv
! mpi_isend
! mpi_iprobe
! mpi_test
! mpi_wait
! mpi_waitall
!
  call mpi_irecv (recvbuf, count, MPI_INTEGER, source, tag, &
                  comm, request, ret)
  call mpi_iprobe (source, tag, comm, flag, status, ret)
  if (flag) then
    write(6,*) "iam=", iam, " mpi_iprobe returns flag=.true. when it should be false"
    call mpi_abort (MPI_COMM_WORLD, -1, ret)
  end if
  call mpi_test (request, flag, status, ret)
  if (flag) then
    write(6,*) "iam=", iam, " mpi_test returns flag=.true. when it should be false"
    call mpi_abort (MPI_COMM_WORLD, -1, ret)
  end if
  call mpi_isend (sendbuf, count, MPI_INTEGER, dest, tag, &
                  comm, request, ret)
  call mpi_wait (request, status, ret)
  call chkbuf ("mpi_wait", recvbuf(:), count, source)

  call mpi_irecv (recvbuf, count, MPI_INTEGER, source, tag, &
                  comm, request, ret)
  call mpi_isend (sendbuf, count, MPI_INTEGER, dest, tag, &
                  comm, request, ret)
  call mpi_waitall (1, request, status, ret)
  call mpi_iprobe (source, tag, comm, flag, status, ret)
  if (.not. flag) then
    write(6,*) "iam=", iam, " mpi_iprobe returns flag=.false. when it should be true"
    call mpi_abort (MPI_COMM_WORLD, -1, ret)
  end if
  call mpi_test (request, flag, status, ret)
  if (.not. flag) then
    write(6,*) "iam=", iam, " mpi_test returns flag=.false. when it should be true"
    call mpi_abort (MPI_COMM_WORLD, -1, ret)
  end if
  call chkbuf ("mpi_waitall", recvbuf(:), count, source)

  call mpi_barrier (comm, ret)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing bcast...'
  end if
!
! mpi_bcast
!
  call mpi_bcast (sendbuf, count, MPI_INTEGER, 0, comm, ret)
  call chkbuf ("MPI_Bcast", sendbuf(:), count, 0)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing allreduce...'
  end if
!
! mpi_allreduce
!
  do i=0,count-1
    sendbuf(i) = iam
  end do

  call mpi_allreduce (sendbuf, recvbuf, count, MPI_INTEGER, MPI_SUM, comm, ret)
  sum = 0.
  do i=0,commsize-1
    sum = sum + i
  end do
  call chkbuf ("mpi_allreduce", recvbuf(:), count, sum)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing gather...'
  end if

  allocate (gsbuf(0:count-1,0:commsize-1))
  allocate (recvcounts(0:commsize-1))
  allocate (sendcounts(0:commsize-1))
!
! mpi_gather
!
  call mpi_gather (sendbuf, count, MPI_INTEGER, &
                   gsbuf, count, MPI_INTEGER, 0, comm, ret)
  if (iam == 0) then
    do j=0,commsize-1
      do i=0,count-1
        if (gsbuf(i,j) /= j) then
          write(6,*) "iam=", iam, "mpi_gather: bad gsbuf(", i,",",j,")=", &
                     gsbuf(i,j), " not= ",j
          call mpi_abort (MPI_COMM_WORLD, -1, ret)
        end if
      end do
    end do
  end if
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing gatherv...'
  end if
!
! mpi_gatherv: make just like mpi_gather for simplicity
!
  gsbuf(:,:) = 0
  recvcounts(:) = count
  rdispls(0) = 0
  do i=1,commsize-1
    rdispls(i) = rdispls(i-1) + recvcounts(i-1)
  end do
  call mpi_gatherv (sendbuf, count, MPI_INTEGER, &
                    gsbuf, recvcounts, rdispls, &
                    MPI_INTEGER, 0, comm, ret)
  if (iam == 0) then
    do j=0,commsize-1
      do i=0,count-1
        if (gsbuf(i,j) /= j) then
          write(6,*) "iam=", iam, "mpi_gatherv: bad gsbuf(", i,",",j,")=", &
                     gsbuf(i,j), " not= ",j
          call mpi_abort (MPI_COMM_WORLD, -1, ret)
        end if
      end do
    end do
  end if
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing scatter...'
  end if
!
! mpi_scatter
!
  call mpi_scatter (gsbuf, count, MPI_INTEGER, recvbuf, count, MPI_INTEGER, 0, comm, ret)
  call chkbuf ("MPI_Scatter", recvbuf(:), count, iam)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing scatterv...'
  end if

  allocate (atoabufsend(0:commsize-1))
  allocate (atoabufrecv(0:commsize-1))
  do i=0,commsize-1
    atoabufsend(i) = i
  end do
!
! mpi_scatterv: make just like mpi_scatter for simplicity.
!
  gsbuf(:,:) = 0
  sendcounts(:) = count
  sdispls(0) = 0
  do i=1,commsize-1
    sdispls(i) = sdispls(i-1) + sendcounts(i-1)
  end do
  call mpi_scatterv (sendbuf, sendcounts, sdispls, &
                     MPI_INTEGER, gsbuf, count, &
                     MPI_INTEGER, 0, comm, ret)
  do i=0,count-1
    if (gsbuf(i,iam) /= iam) then
      write(6,*) "iam=", iam, "mpi_scatterv: bad gsbuf(", i,",",iam,")=", &
           gsbuf(i,j), " not= ",iam
      call mpi_abort (MPI_COMM_WORLD, -1, ret)
    end if
  end do
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing alltoall...'
  end if
!
! mpi_alltoall
!
  call mpi_alltoall (atoabufsend, 1, MPI_INTEGER, atoabufrecv, 1, MPI_INTEGER, comm, ret)
  do i=0,commsize-1
    if (atoabufrecv(i) /= iam) then
      write(6,*) "iam=", iam, "mpi_alltoall: bad atoabufrecv(",i,")=",atoabufrecv(i)
      call mpi_abort (MPI_COMM_WORLD, -1, ret)
    end if
  end do
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing alltoallv...'
  end if
!
! mpi_alltoallv
!
  do j=0,commsize-1
    if (j == iam) then
      gsbuf(:,j) = j
    else
      gsbuf(:,j) = 0
    end if
  end do
  call mpi_alltoallv (gsbuf, sendcounts, sdispls, MPI_INTEGER, &
                      gsbuf, recvcounts, rdispls, MPI_INTEGER, comm, ret)
  do j=0,commsize-1
    do i=0,count-1
      if (gsbuf(i,j) /= j) then
        write(6,*) "iam=", iam, "mpi_alltoallv: bad gsbuf(", i,",",j,")=", &
             gsbuf(i,j), " not= ",j
        call mpi_abort (MPI_COMM_WORLD, -1, ret)
      end if
    end do
  end do
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing reduce...'
  end if
!
! mpi_reduce
!
  call mpi_reduce (sendbuf, recvbuf, count, MPI_INTEGER, MPI_SUM, 0, comm, ret)
  if (iam == 0) then
    sum = 0.
    do i=0,commsize-1
      sum = sum + i
    end do
    call chkbuf ("mpi_reduce", recvbuf(:), count, sum)
  end if
!
! mpi_allgather
!
  gsbuf(:,:) = 0
  call mpi_allgather (sendbuf, count, MPI_INTEGER, &
                      gsbuf, count, MPI_INTEGER, comm, ret)
  do j=0,commsize-1
    do i=0,count-1
      if (gsbuf(i,j) /= j) then
        write(6,*) "iam=", iam, "mpi_allgather: bad gsbuf(",i,",",j,")=", &
                    gsbuf(i,j)
        call mpi_abort (MPI_COMM_WORLD, -1, ret)
      end if
    end do
  end do
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing allgatherv...'
  end if
!
! mpi_allgatherv: Make just like mpi_allgather for simplicity
!
  gsbuf(:,:) = 0
  recvcounts(:) = count
  call mpi_allgatherv (sendbuf, count, MPI_INTEGER, &
                       gsbuf, recvcounts, rdispls, &
                       MPI_INTEGER, comm, ret)
  do j=0,commsize-1
    do i=0,count-1
      if (gsbuf(i,j) /= j) then
        write(6,*) "iam=", iam, "mpi_allgatherv: bad gsbuf(",i,",",j,")=", &
                    gsbuf(i,j)
        call mpi_abort (MPI_COMM_WORLD, -1, ret)
      end if
    end do
  end do
!
! mpi_finalize
!
  call mpi_finalize (ret)

#if ( ! defined HAVE_IARGCGETARG )
  ret = gptlstop ("total")
  ret = gptlpr (iam)
#endif

  stop 0
end program pmpi

subroutine chkbuf (msg, recvbuf, count, source)
  use myvars
  implicit none

#include <mpif.h>

  character(len=*), intent(in) :: msg

  integer, intent(in) :: count
  integer, intent(in) :: recvbuf(0:count-1)
  integer, intent(in) :: source

  integer :: i
  integer :: ret
  do i=0,count-1
    if (recvbuf(i) /= source) then
      write(6,*) "iam=", iam, msg, "bad recvbuf(", i,")=",recvbuf(i), "/= ", source
      call mpi_abort (MPI_COMM_WORLD, -1, ret)
    end if
  end do
end subroutine chkbuf
