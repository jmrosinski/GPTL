module inplace_subs
  ! Routines to test use of MPI_IN_PLACE. This is NOT the exhaustive list of MPI routines, only
  ! the exhaustive list of GPTL Fortran routines which implement the pmpi interface.
  use mpi
  implicit none

  private
  public :: reduce, allreduce, gather, gatherv, scatter, scatterv, allgather, allgatherv, &
            alltoall, alltoallv
  integer :: ierr
  integer, parameter :: spval = -999
  
CONTAINS

  function reduce (myid, numprocs) result (ret)
    integer, intent(in) :: myid, numprocs
    
    character(len=*), parameter :: thisfunc = ' reduce'
    integer :: ret
    integer :: sendbuf
    integer :: recvbuf

    if (myid == 0) then
      sendbuf = MPI_IN_PLACE
      recvbuf = myid
    else
      sendbuf = myid
    end if
    call MPI_Reduce (sendbuf, recvbuf, 1, MPI_INTEGER, MPI_MAX, 0, MPI_COMM_WORLD, ret)

    if (myid == 0) then
      ret = chkbuf (ret, myid, thisfunc, recvbuf, numprocs-1) ! correct MPI_IN_PLACE
    end if

    call MPI_Barrier (MPI_COMM_WORLD, ierr)
  end function reduce
  
  function allreduce (myid, numprocs) result (ret)
    integer, intent(in) :: myid, numprocs
    
    character(len=*), parameter :: thisfunc = ' allreduce'    
    integer :: ret
    integer :: sendbuf
    integer :: recvbuf

    if (myid == 0) then
      sendbuf = MPI_IN_PLACE
      recvbuf = myid
    else
      sendbuf = myid
    end if
    call MPI_Allreduce (sendbuf, recvbuf, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ret)

    ret = chkbuf (ret, myid, thisfunc, recvbuf, numprocs-1)
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
  end function allreduce

  function gather (myid, numprocs) result (ret)
    integer, intent(in) :: myid, numprocs
    
    character(len=*), parameter :: thisfunc = ' gather'
    integer :: n, ret
    integer :: sendbuf                  ! scalar ok since only sending 1 value
    integer :: recvbuf(0:numprocs-1)

    recvbuf(:) = -1
    if (myid == 0) then
      sendbuf = MPI_IN_PLACE
      recvbuf(myid) = myid
    else
      sendbuf = myid
    end if
    call MPI_Gather (sendbuf, 1, MPI_INTEGER, &
                     recvbuf, 1, MPI_INTEGER, &
                     0, MPI_COMM_WORLD, ret)
    if (myid == 0) then
      do n=0,numprocs-1
        if (chkbuf (ret, myid, thisfunc, recvbuf(n), n) /= 0) then
          ret = 1
        end if
      end do
    end if
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
  end function gather
  
  function gatherv (myid, numprocs) result (ret)
    integer, intent(in) :: myid, numprocs
    
    character(len=*), parameter :: thisfunc = ' gatherv'
    integer :: n, ret
    integer :: sendbuf                  ! scalar ok since only sending 1 value
    integer :: recvbuf(0:numprocs-1)
    integer :: recvcounts(0:numprocs-1)
    integer :: displs(0:numprocs-1)

    recvbuf(:)    = -1
    recvcounts(:) = 1
    displs(:) = (/(n,n=0,numprocs-1)/)
    if (myid == 0) then
      sendbuf = MPI_IN_PLACE
      recvbuf(myid) = myid
    else
      sendbuf = myid
    end if
    call MPI_Gatherv (sendbuf, 1,                  MPI_INTEGER, &
                      recvbuf, recvcounts, displs, MPI_INTEGER, &
                      0, MPI_COMM_WORLD, ret)
    if (myid == 0) then
      do n=0,numprocs-1
        if (chkbuf (ret, myid, thisfunc, recvbuf(n), n) /= 0) then
          ret = 1
        end if
      end do
    end if
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
  end function gatherv
  
  function scatter (myid, numprocs) result (ret)
    integer, intent(in) :: myid, numprocs
    
    character(len=*), parameter :: thisfunc = ' scatter'
    integer :: n, ret
    integer :: sendbuf(0:numprocs-1)
    integer :: recvbuf               ! scalar ok since only sending 1 value

    if (myid == 0) then
      sendbuf(:) = (/(n,n=0,numprocs-1)/)
      recvbuf = spval   ! WRONG value to verify was kept in-place
      call MPI_Scatter (sendbuf,      1, MPI_INTEGER, &
                        MPI_IN_PLACE, 1, MPI_INTEGER, & ! root ignores recvcount, recvtype
                        0, MPI_COMM_WORLD, ret)
    else
      recvbuf = -1
      call MPI_Scatter (sendbuf,       1, MPI_INTEGER, &
                        recvbuf,       1, MPI_INTEGER, &
                        0, MPI_COMM_WORLD, ret)
    end if

    if (myid == 0) then
      ret = chkbuf (ret, myid, thisfunc, recvbuf, spval) ! Verify spval kept due to MPI_IN_PLACE
    else
      ret = chkbuf (ret, myid, thisfunc, recvbuf, myid)
    end if
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
  end function scatter
  
  function scatterv (myid, numprocs) result (ret)
    integer, intent(in) :: myid, numprocs
    
    character(len=*), parameter :: thisfunc = ' scatterv'
    integer :: n, ret
    integer :: sendbuf(0:numprocs-1)
    integer :: sendcounts(0:numprocs-1)
    integer :: displs(0:numprocs-1)
    integer :: recvbuf               ! scalar ok since only sending 1 value

    sendcounts(:) = 1
    displs(:)     = (/(n,n=0,numprocs-1)/)
    if (myid == 0) then
      sendbuf(:) = (/(n,n=0,numprocs-1)/)
      recvbuf = spval   ! WRONG value to verify was kept in-place
      call MPI_Scatterv (sendbuf,        sendcounts, displs, MPI_INTEGER, &
                         MPI_IN_PLACE,   1,                  MPI_INTEGER, &
                         0, MPI_COMM_WORLD, ret)
    else
      recvbuf = -1
      call MPI_Scatterv (sendbuf,        sendcounts, displs, MPI_INTEGER, &
                         recvbuf,        1,                  MPI_INTEGER, &
                         0, MPI_COMM_WORLD, ret)
    end if

    if (myid == 0) then
      ret = chkbuf (ret, myid, thisfunc, recvbuf, spval) ! Verify spval kept due to MPI_IN_PLACE
    else
      ret = chkbuf (ret, myid, thisfunc, recvbuf, myid)
    end if
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
  end function scatterv

  function allgather (myid, numprocs) result (ret)
    integer, intent(in) :: myid, numprocs
    
    character(len=*), parameter :: thisfunc = ' allgather'
    integer :: n, ret
    integer :: recvbuf(0:numprocs-1)

    recvbuf(:) = -1
    recvbuf(myid) = myid
    call MPI_Allgather (MPI_IN_PLACE, 1, MPI_INTEGER, &
                        recvbuf,      1, MPI_INTEGER, &
                        MPI_COMM_WORLD, ret)
    do n=0,numprocs-1
      if (chkbuf (ret, myid, thisfunc, recvbuf(n), n) /= 0) then
        ret = 1
      end if
    end do
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
  end function allgather
  
  function allgatherv (myid, numprocs) result (ret)
    integer, intent(in) :: myid, numprocs
    
    character(len=*), parameter :: thisfunc = ' allgatherv'
    integer :: n, ret
    integer :: recvbuf(0:numprocs-1)
    integer :: recvcounts(0:numprocs-1)
    integer :: displs(0:numprocs-1)

    recvbuf(:) = -1
    recvbuf(myid) = myid
    recvcounts(:) = 1
    displs(:) = (/(n,n=0,numprocs-1)/)
    call MPI_Allgatherv (MPI_IN_PLACE, 1,                  MPI_INTEGER, &
                         recvbuf,      recvcounts, displs, MPI_INTEGER, &
                         MPI_COMM_WORLD, ret)
    do n=0,numprocs-1
      if (chkbuf (ret, myid, thisfunc, recvbuf(n), n) /= 0) then
        ret = 1
      end if
    end do
    call MPI_Barrier (MPI_COMM_WORLD, ierr)
  end function allgatherv
  
  function chkbuf (inret, rank, func, got, shouldbe) result (outret)
    integer, intent(in) :: inret
    integer, intent(in) :: rank
    character(len=*), intent(in) :: func
    integer, intent(in) :: got
    integer, intent(in) :: shouldbe

    integer :: outret

    outret = 0
    if (inret /= 0) then
      write(6,*) func, ' rank ', rank, ' failure'
      outret = inret
      return
    end if

    if (got == shouldbe) then
      write(6,*) func, ' rank ', rank, ' success'
    else
      outret = -1
      write(6,*) func, ' rank ', rank, ' failure got ', got, ' should have got ', shouldbe
    end if
    return
  end function chkbuf

  ! Calling any of the alltoall functions should result in MPI_Abort
  integer function alltoall (myid, numprocs)
    integer, intent(in) :: myid, numprocs
    
    integer :: ret
    integer :: sbuf(numprocs)
    integer :: rbuf(numprocs)
    
    call MPI_Alltoall (MPI_IN_PLACE, 1, MPI_INTEGER, &
                       rbuf,         1, MPI_INTEGER, MPI_COMM_WORLD, ret)
    alltoall = -1
  end function alltoall

  integer function alltoallv (myid, numprocs)
    integer, intent(in) :: myid, numprocs
    
    integer :: ret
    integer :: sbuf(numprocs)
    integer :: scounts(numprocs)
    integer :: sdispls(numprocs)
    integer :: rbuf(numprocs)
    integer :: rcounts(numprocs)
    integer :: rdispls(numprocs)
    
    call MPI_Alltoallv (MPI_IN_PLACE, scounts, sdispls, MPI_INTEGER, &
		        rbuf,         rcounts, rdispls, MPI_INTEGER, MPI_COMM_WORLD, ret);
    alltoallv = -1
  end function alltoallv
end module inplace_subs
