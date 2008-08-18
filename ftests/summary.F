      program main 
#include <mpif.h>
#include "../gptl.inc"
      double precision, external :: sub
      double precision result

      integer :: nthreads = 1 ! number of threads (default 1)
      integer :: nproc = 1
      integer iter
      integer papiopt
      integer c
      integer :: comm = 0
      integer ierr
      integer ret

#ifdef HAVE_MPI
      call mpi_init (ierr)
      comm = MPI_COMM_WORLD
#endif

      ret = gptlsetoption (gptlabort_on_error, 1)
      ret = gptlsetoption (gptloverhead, 1)
      ret = gptlsetoption (gptlnarrowprint, 1)

      ret = gptlinitialize ()
      ret = gptlstart ("total")
	 
#ifdef HAVE_MPI
      call mpi_comm_rank (MPI_COMM_WORLD, iam, ierr)
      call mpi_comm_size (MPI_COMM_WORLD, nproc, ierr)
#endif
      if (iam == 0) then
        write (6,*) "Purpose: test behavior of summary stats"
        write (6,*) "Include OpenMP if enabled"
      end if

#ifdef THEADED_OMP
      nthreads = omp_get_max_threads ()
#endif

!$OMP PARALLEL DO PRIVATE (ITER, RESULT)
      do iter=1,nthreads
        result = sub (iter, iam)
      end do

      ret = gptlstop ("total")
      ret = gptlpr (iam)
      ret = gptlpr_summary (comm)
      if (gptlfinalize () < 0) stop 6

#ifdef HAVE_MPI
      call mpi_finalize ()
#endif
      stop 0
      end program main



      double precision function sub (iter, iam)
      integer, intent (in) :: iter
      integer, intent (in) :: iam

      integer (8) :: looplen
      integer (8) :: i
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
