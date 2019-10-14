program main 
#ifdef THREADED_OMP
  use omp_lib
#endif
#ifdef HAVE_LIBMPI
  use mpi
#endif
  use gptl

  implicit none
  double precision, external :: sub
  double precision result
  external :: checkstat

  integer :: iam = 0
  integer :: nthreads = 1 ! number of threads (default 1)
  integer :: nproc = 1
  integer iter
  integer code
  integer c
  integer :: comm = 0
  integer ierr
  integer ret
  character(len=8), parameter :: prognam = 'summary'

#ifdef HAVE_PAPI
! Turn abort_on_error off just long enough to check PAPI-based options
  ret = gptlsetoption (gptlabort_on_error, 0)
  if (gptlevent_name_to_code ('PAPI_TOT_CYC', code) == 0) then
    ret = gptlsetoption (code, 1)
  end if
  ret = gptlsetoption (gptlabort_on_error, 1)
#endif

  ret = gptlsetoption (gptlabort_on_error, 1)
  call checkstat (ret, prognam//': Error from gptlsetoption(gptlabort_on_error,1)')
  ret = gptlsetoption (gptloverhead, 1)
  call checkstat (ret, prognam//': Error from gptlsetoption(gptloverhead,1)')
  ret = gptlsetoption (gptlnarrowprint, 1)
  call checkstat (ret, prognam//': Error from gptlsetoption(gptlnarrowprint,1)')

#ifdef HAVE_LIBMPI
  call mpi_init (ierr)
  comm = MPI_COMM_WORLD
  call mpi_comm_rank (MPI_COMM_WORLD, iam, ierr)
  call mpi_comm_size (MPI_COMM_WORLD, nproc, ierr)
#endif

  ret = gptlinitialize ()
  call checkstat (ret, prognam//': Error from gptlinitialize()')
  ret = gptlstart ("total")

  if (iam == 0) then
    write (6,*) "Purpose: test behavior of summary stats"
    write (6,*) "Include OpenMP if enabled"
  end if

#ifdef THREADED_OMP
  nthreads = omp_get_max_threads ()
#endif

!$OMP PARALLEL DO PRIVATE (RESULT)
  do iter=1,nthreads
    result = sub (iter, iam)
  end do

  ret = gptlstop ("total")
  ret = gptlpr (iam)
  call checkstat (ret, prognam//': Error from gptlpr(iam)')
  ret = gptlpr_summary (comm)
  call checkstat (ret, prognam//': Error from gptlpr_summary(comm)')

#ifdef HAVE_LIBMPI
  call mpi_finalize (ret)
#endif
  
  ret = gptlfinalize ()
  call checkstat (ret, prognam//': Error from gptlfinalize()')
  stop 0
end program main


double precision function sub (iter, iam)
  use gptl
  implicit none
  
  integer, intent (in) :: iter
  integer, intent (in) :: iam

  integer (8) :: looplen
  integer (8) :: i
  integer :: ret
  double precision sum

  looplen = iam*iter*10000
  ret = gptlstart ("sub")

  ret = gptlstart ("sleep")
  ret = gptlstop ("sleep")

  ret = gptlstart ("work")
  sum = 0.
  ret = gptlstart ("add")
  do i=0,looplen-1
    sum = sum + i
  end do
  ret = gptlstop ("add")

  ret = gptlstart ("madd")
  do i=0,looplen-1
    sum = sum + i*1.1
  end do
  ret = gptlstop ("madd")
  
  ret = gptlstart ("div")
  do i=0,looplen-1
    sum = sum / 1.1
  end do
  ret = gptlstop ("div")
  ret = gptlstop ("work")
  ret = gptlstop ("sub")
  
  sub = sum
  return 
end function sub

subroutine checkstat (ret, str)
  implicit none

  integer, intent(in) :: ret
  character(len=*), intent(in) :: str

  if (ret /= 0) then
    write(6,*) 'Bad return code=', ret, ' ', str
    stop 1
  end if
end subroutine checkstat
