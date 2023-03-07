function get_f_mpi_in_place (f_mpi_in_place) result(ier)
!
! get_f_mpi_in_place.F90
!
! Author: Jim Rosinski
!
! Utility function called from C returns the address of Fortran version of MPI variable
! MPI_IN_PLACE. Required when PMPI profiling enabled and one or more MPI routines which
! allow MPI_IN_PLACE for "sendbuf" or "recvbuf" 
!  
  use mpi
  implicit none
  integer(KIND=MPI_ADDRESS_KIND), intent(out) :: f_mpi_in_place
  integer :: ier

  call mpi_get_address (mpi_in_place, f_mpi_in_place, ier)
  if (ier /= MPI_SUCCESS) then
    write(6,*)'get_f_mpi_in_place: failure from mpi_get_address'
  end if
end function get_f_mpi_in_place
