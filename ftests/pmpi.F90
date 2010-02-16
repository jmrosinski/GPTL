module myvars
  integer :: iam
  integer :: commsize
end module myvars

program pmpi
  use myvars
  implicit none

#include <mpif.h>
#include "../gptl.inc"

  integer, parameter :: tag = 98
  integer, parameter :: count = 1024

  integer :: i, j, ret
  integer :: val
  integer :: comm = MPI_COMM_WORLD
  integer :: sendbuf(0:count-1)
  integer :: recvbuf(0:count-1)
  integer :: sum
  integer :: status(MPI_STATUS_SIZE)
  integer :: sendreq, recvreq
  integer :: dest
  integer :: source
  integer :: rdispls(0:count-1)
  integer :: sdispls(0:count-1)

  integer, allocatable :: atoabufsend(:)
  integer, allocatable :: atoabufrecv(:)
  integer, allocatable :: gsbufsend(:,:)      ! gather/scatter buffer send
  integer, allocatable :: gsbufrecv(:,:)      ! gather/scatter buffer recv
  integer, allocatable :: recvcounts(:)
  integer, allocatable :: sendcounts(:)
  integer, allocatable :: atoacounts(:)
  integer, allocatable :: atoadispls(:)

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

  write(0,*)'Calling mpi_init'
  call mpi_init (ret)
  write(0,*)'Calling setlinebuf_stdout'
!  call setlinebuf_stdout ()

! For debugging, go into infinite loop so debugger can attach and reset
#ifdef DEBUG
  do while (debugflag == 1)
  end do
#endif
  write(0,*)'Calling mpi_comm_rank'
  call mpi_comm_rank (comm, iam, ret)
  write(0,*)'Hello from rank ', iam
  write(0,*)'iam=',iam,'Calling mpi_comm_size'
  call mpi_comm_size (comm, commsize, ret)
  write(0,*)'iam=',iam,'commsize is ', commsize
  call flush()

!  stop 0

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
  recvbuf(:) = -1
  write(6,*)'iam=',iam,'Starting first mpi loop'
  if (mod (commsize, 2) == 0) then
    if (iam == 0) then
      write(6,*)'pmpi.F90: testing send, recv, probe...'
    end if

    if (mod (iam, 2) == 0) then
      write(6,*)'iam=',iam,'even calling send'
      call flush()
      call mpi_send (sendbuf, count, MPI_INTEGER, dest, tag, comm, ret)
      write(6,*)'iam=',iam,'even calling recv'
      call flush()
      call mpi_recv (recvbuf, count, MPI_INTEGER, source, tag, comm, status, ret)
    else
      write(6,*)'iam=',iam,'odd calling probe'
      call flush()
      call mpi_probe (source, tag, comm, status, ret)
      if (ret /= MPI_SUCCESS) then
        write(6,*) "iam=", iam, " mpi_probe: bad return"
        call mpi_abort (MPI_COMM_WORLD, -1, ret)
      end if
      write(6,*)'iam=',iam,'odd calling recv'
      call flush()
      call mpi_recv (recvbuf, count, MPI_INTEGER, source, tag, comm, status, ret)
      write(6,*)'iam=',iam,'odd calling send'
      call flush()
      call mpi_send (sendbuf, count, MPI_INTEGER, dest, tag, comm, ret)
    end if
    call chkbuf ('mpi_send + mpi_recv', recvbuf(:), count, source)

    if (iam == 0) then
      write(6,*)'Success'
      write(6,*)'pmpi.F90: testing ssend...'
      call flush()
    end if
!
! mpi_ssend
!
    recvbuf(:) = -1
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
      call flush()
    end if
  end if
!
! mpi_sendrecv
! 
  recvbuf(:) = -1
  call mpi_sendrecv (sendbuf, count, MPI_INTEGER, dest, tag, &
                     recvbuf, count, MPI_INTEGER, source, tag, &
                     comm, status, ret)
  call chkbuf ('mpi_sendrecv', recvbuf(:), count, source)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing irecv, isend, iprobe, itest, wait, waitall...'
    call flush()
  end if
!
! mpi_irecv
! mpi_isend
! mpi_iprobe
! mpi_test
! mpi_wait
! mpi_waitall
!
  write(6,*)'iam=',iam,' calling irecv...'
  call flush()
  recvbuf(:) = -1
  call mpi_irecv (recvbuf, count, MPI_INTEGER, source, tag, &
                  comm, recvreq, ret)
  write(6,*)'iam=',iam,'calling iprobe...'
  call flush()
  call mpi_iprobe (source, tag, comm, flag, status, ret)
  write(6,*)'iam=',iam,'calling test...'
  call flush()
  call mpi_test (recvreq, flag, status, ret)
  write(6,*)'iam=',iam,'calling isend...'
  call flush()
  call mpi_isend (sendbuf, count, MPI_INTEGER, dest, tag, &
                  comm, sendreq, ret)
  write(6,*)'iam=',iam,'calling wait...'
  call flush()
  call mpi_wait (recvreq, status, ret)
  call chkbuf ("mpi_wait", recvbuf(:), count, source)

  write(6,*)'iam=',iam,'calling irecv 2nd time...'
  call flush()
  recvbuf(:) = -1
  call mpi_irecv (recvbuf, count, MPI_INTEGER, source, tag, &
                  comm, recvreq, ret)
  write(6,*)'iam=',iam,'calling isend 2nd time...'
  call flush()
  call mpi_isend (sendbuf, count, MPI_INTEGER, dest, tag, &
                  comm, sendreq, ret)
  write(6,*)'iam=',iam,'calling waitall...'
  call flush()
  call mpi_waitall (1, recvreq, status, ret)
  call chkbuf ("mpi_waitall", recvbuf(:), count, source)

  call mpi_barrier (comm, ret)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing bcast...'
    call flush()
  end if
!
! mpi_bcast
!
  call mpi_bcast (sendbuf, count, MPI_INTEGER, 0, comm, ret)
  call chkbuf ("mpi_bcast", sendbuf(:), count, 0)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing allreduce...'
    call flush()
  end if
!
! mpi_allreduce: need to reset sendbuf due to bcast just done
!
  do i=0,count-1
    sendbuf(i) = iam
  end do

  recvbuf(:) = -1
  call mpi_allreduce (sendbuf, recvbuf, count, MPI_INTEGER, MPI_SUM, comm, ret)
  sum = 0.
  do i=0,commsize-1
    sum = sum + i
  end do
  call chkbuf ("mpi_allreduce", recvbuf(:), count, sum)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing gather...'
    call flush()
  end if

  allocate (gsbufsend(0:count-1,0:commsize-1))
  allocate (gsbufrecv(0:count-1,0:commsize-1))
  allocate (recvcounts(0:commsize-1))
  allocate (sendcounts(0:commsize-1))
!
! mpi_gather
!
  gsbufrecv(:,:) = -1
  call mpi_gather (sendbuf, count, MPI_INTEGER, &
                   gsbufrecv, count, MPI_INTEGER, 0, comm, ret)
  if (iam == 0) then
    do j=1,commsize-1
      call chkbuf ("mpi_gather", gsbufrecv(:,j), count, j)
    end do
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing gatherv...'
    call flush()
  end if
!
! mpi_gatherv: make just like mpi_gather for simplicity
!
  gsbufrecv(:,:) = -1
  recvcounts(:) = count
  rdispls(0) = 0
  do j=1,commsize-1
    rdispls(j) = rdispls(j-1) + recvcounts(j-1)
  end do
  call mpi_gatherv (sendbuf, count, MPI_INTEGER, &
                    gsbufrecv, recvcounts, rdispls, &
                    MPI_INTEGER, 0, comm, ret)
  if (iam == 0) then
    do j=1,commsize-1
      call chkbuf ("mpi_gatherv", gsbufrecv(:,j), count, j)
    end do
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing scatter...'
    call flush()
  end if
!
! mpi_scatter
!
  if (iam == 0) then
    do j=0,commsize-1
      gsbufsend(:,j) = j
    end do
  else
    do j=0,commsize-1
      gsbufsend(:,j) = -1
    end do
  end if
  recvbuf(:) = -1
  call mpi_scatter (gsbufsend, count, MPI_INTEGER, recvbuf, count, MPI_INTEGER, &
                    0, comm, ret)
  call chkbuf ("mpi_scatter", recvbuf(:), count, iam)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing scatterv...'
    call flush()
  end if
!
! mpi_scatterv: make just like mpi_scatter for simplicity.
!
  if (iam == 0) then
    do j=0,commsize-1
      gsbufsend(:,j) = j
    end do
  else
    gsbufsend(:,:) = -1
  end if
  sendcounts(:) = count
  sdispls(0) = 0
  do j=1,commsize-1
    sdispls(j) = sdispls(j-1) + sendcounts(j-1)
  end do
  recvbuf(:) = -1
  call mpi_scatterv (gsbufsend, sendcounts, sdispls, &
                     MPI_INTEGER, recvbuf, count, &
                     MPI_INTEGER, 0, comm, ret)
  call chkbuf ("mpi_scatterv", recvbuf(:), count, iam)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing alltoall...'
    call flush()
  end if
!
! mpi_alltoall
!
  allocate (atoabufsend(0:commsize-1))
  allocate (atoabufrecv(0:commsize-1))
  allocate (atoacounts(0:commsize-1))
  allocate (atoadispls(0:commsize-1))
  do j=0,commsize-1
    atoabufsend(j) = j
  end do
  atoabufrecv(:) = -1
  call mpi_alltoall (atoabufsend, 1, MPI_INTEGER, atoabufrecv, 1, MPI_INTEGER, comm, ret)
  call chkbuf ("mpi_alltoall", atoabufrecv(:), 1, iam)
  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing alltoallv...'
    call flush()
  end if
!
! mpi_alltoallv
!
  atoabufrecv(:) = -1
  atoacounts(:) = 1
  atoadispls(0) = 0
  do j=1,commsize-1
    atoadispls(j) = atoadispls(j-1) + atoacounts(j-1)
  end do
  
  call mpi_alltoallv (atoabufsend, atoacounts, atoadispls, MPI_INTEGER, &
                      atoabufrecv, atoacounts, atoadispls, MPI_INTEGER, comm, ret)
  call chkbuf ("mpi_alltoall", atoabufrecv(:), 1, iam)

  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing reduce...'
    call flush()
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

  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing allgather...'
    call flush()
  end if
!
! mpi_allgather
!
  gsbufrecv(:,:) = -1
  call mpi_allgather (sendbuf, count, MPI_INTEGER, &
                      gsbufrecv, count, MPI_INTEGER, comm, ret)
  do j=0,commsize-1
    call chkbuf ("mpi_allgather", gsbufrecv(:,j), count, j)
  end do

  if (iam == 0) then
    write(6,*)'Success'
    write(6,*)'pmpi.F90: testing allgatherv...'
    call flush()
  end if
!
! mpi_allgatherv: Make just like mpi_allgather for simplicity
!
  gsbufrecv(:,:) = -1
  recvcounts(:) = count
  call mpi_allgatherv (sendbuf, count, MPI_INTEGER, &
                       gsbufrecv, recvcounts, rdispls, &
                       MPI_INTEGER, comm, ret)
  do j=0,commsize-1
    call chkbuf ("mpi_allgatherv", gsbufrecv(:,j), count, j)
  end do

  if (iam == 0) then
    write(6,*)'Success. Calling finalize'
    call flush()
  end if
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

subroutine chkbuf (msg, recvbuf, count, val)
  use myvars
  implicit none

#include <mpif.h>

  character(len=*), intent(in) :: msg

  integer, intent(in) :: count
  integer, intent(in) :: recvbuf(0:count-1)
  integer, intent(in) :: val

  integer :: i
  integer :: ret
  do i=0,count-1
    if (recvbuf(i) /= val) then
      write(6,*) "iam=", iam, msg, " bad recvbuf(", i,")=",recvbuf(i), "/= ", val
      call mpi_abort (MPI_COMM_WORLD, -1, ret)
    end if
  end do
end subroutine chkbuf
