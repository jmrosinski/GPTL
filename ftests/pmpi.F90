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

  integer, allocatable :: atoabufsend(:)
  integer, allocatable :: atoabufrecv(:)
  integer, allocatable :: gsbuf(:,:)

  ret = gptlsetoption (gptloverhead, 0)
  ret = gptlsetoption (gptlpercent, 0)
  ret = gptlsetoption (gptlabort_on_error, 1)
  ret = gptlsetoption (gptlsync_mpi, 1)

#if ( ! defined HAVE_IARGCGETARG )
  ret = gptlinitialize ()
  ret = gptlstart ("total")
#endif

  call mpi_init (ret)
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

  if (mod (commsize, 2) == 0) then
    if (mod (iam, 2) == 0) then
      call mpi_send (sendbuf, count, MPI_INTEGER, dest, tag, comm, ret)
      call mpi_recv (recvbuf, count, MPI_INTEGER, source, tag, comm, status, ret)
    else
      call mpi_recv (recvbuf, count, MPI_INTEGER, source, tag, comm, status, ret)
      call mpi_send (sendbuf, count, MPI_INTEGER, dest, tag, comm, ret)
    end if
  end if
  call chkbuf ('mpi_send + mpi_recv', recvbuf(:), count, source)
  
  call mpi_sendrecv (sendbuf, count, MPI_INTEGER, dest, tag, &
                     recvbuf, count, MPI_INTEGER, source, tag, &
                     comm, status, ret)
  call chkbuf ('mpi_sendrecv', recvbuf(:), count, source)

  call mpi_irecv (recvbuf, count, MPI_INTEGER, source, tag, &
                  comm, request, ret)
  call mpi_isend (sendbuf, count, MPI_INTEGER, dest, tag, &
                  comm, request, ret)
  call mpi_wait (request, status, ret)
  call chkbuf ("mpi_wait", recvbuf(:), count, source)

  call mpi_irecv (recvbuf, count, MPI_INTEGER, source, tag, &
                  comm, request, ret)
  call mpi_isend (sendbuf, count, MPI_INTEGER, dest, tag, &
                  comm, request, ret)
  call mpi_waitall (1, request, status, ret)
  call chkbuf ("mpi_waitall", recvbuf(:), count, source)

  call mpi_barrier (comm, ret)

  call mpi_bcast (sendbuf, count, MPI_INTEGER, 0, comm, ret)
  call chkbuf ("MPI_Bcast", sendbuf(:), count, 0)

  do i=0,count-1
    sendbuf(i) = iam
  end do

  call mpi_allreduce (sendbuf, recvbuf, count, MPI_INTEGER, MPI_SUM, comm, ret)
  sum = 0.
  do i=0,commsize-1
    sum = sum + i
  end do
  call chkbuf ("mpi_allreduce", recvbuf(:), count, sum)

  allocate (gsbuf(0:count-1,0:commsize-1))
      
  call mpi_gather (sendbuf, count, MPI_INTEGER, gsbuf, count, MPI_INTEGER, 0, comm, ret)
  if (iam == 0) then
    do j=0,commsize-1
      do i=0,count-1
        if (gsbuf(i,j) /= j) then
          write(6,*) "iam=", iam, "MPI_Gather: bad gsbuf(", i,",",j,")=", &
                     gsbuf(i,j), " not= ",j
          call mpi_abort (MPI_COMM_WORLD, -1, ret)
        end if
      end do
    end do
  end if

  call mpi_scatter (gsbuf, count, MPI_INTEGER, recvbuf, count, MPI_INTEGER, 0, comm, ret)
  call chkbuf ("MPI_Scatter", recvbuf(:), count, iam)

  allocate (atoabufsend(0:commsize-1))
  allocate (atoabufrecv(0:commsize-1))
  do i=0,commsize-1
    atoabufsend(i) = i
  end do

  call mpi_alltoall (atoabufsend, 1, MPI_INTEGER, atoabufrecv, 1, MPI_INTEGER, comm, ret)
  do i=0,commsize-1
    if (atoabufrecv(i) /= iam) then
      write(6,*) "iam=", iam, "MPI_Alltoall: bad atoabufrecv(",i,")=",atoabufrecv(i)
      call mpi_abort (MPI_COMM_WORLD, -1, ret)
    end if
  end do

  call mpi_reduce (sendbuf, recvbuf, count, MPI_INTEGER, MPI_SUM, 0, comm, ret)
  if (iam == 0) then
    sum = 0.
    do i=0,commsize-1
      sum = sum + i
    end do
    call chkbuf ("mpi_reduce", recvbuf(:), count, sum)
  end if

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
