      program legacy
      include 'gptl.inc'
#ifdef HAVE_LIBMPI
      include 'mpif.h'
      include 'gptlmpi.inc'
#endif
      integer ret, ierr

#ifdef HAVE_LIBMPI
      call mpi_init (ierr)
      if (ierr .ne. 0) then
        write(6,*) 'Failure from mpi_init'
        call exit(1)
      end if
#endif
      if (gptlinitialize() .ne. 0) then
        write(6,*) 'Failure from gptlinitialize'
        call exit(1)
      end if
      if (gptlstart ('main') .ne. 0) then
        write(6,*) 'Failure from gptlstart'
        call exit(1)
      end if
      if (gptlstop ('main') .ne. 0) then
        write(6,*) 'Failure from gptlstop'
        call exit(1)
      end if
      if (gptlpr(0) .ne. 0) then
        write(6,*) 'Failure from gptlpr'
        call exit(1)
      end if
#ifdef HAVE_LIBMPI
      if (gptlpr_summary(MPI_COMM_WORLD) .ne. 0) then
        write(6,*) 'Failure from gptlpr_summary'
        call exit(1)
      end if
#endif
      write (6,*)'Success'
      write (6,*)'legacy: All tests succeeded'
      end
