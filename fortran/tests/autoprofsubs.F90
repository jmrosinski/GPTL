subroutine innersub1 (inner, sum)
  implicit none
  integer, intent(in) :: inner
  double precision, intent(inout) :: sum
  integer :: j
  do j=0,inner-1
    sum = sum + j
  end do
end subroutine innersub1

subroutine innersub10 (inner, sum)
  implicit none
  integer, intent(in) :: inner
  double precision, intent(inout) :: sum
  integer :: j
  do j=0,inner-1
    sum = sum + j
  end do
end subroutine innersub10

subroutine innersub100 (inner, sum)
  implicit none
  integer, intent(in) :: inner
  double precision, intent(inout) :: sum
  integer :: j
  do j=0,inner-1
    sum = sum + j
  end do
end subroutine innersub100

subroutine innersub1000 (inner, sum)
  implicit none
  integer, intent(in) :: inner
  double precision, intent(inout) :: sum
  integer :: j
  do j=0,inner-1
    sum = sum + j
  end do
end subroutine innersub1000

subroutine innersub10000 (inner, sum)
  implicit none
  integer, intent(in) :: inner
  double precision, intent(inout) :: sum
  integer :: j
  do j=0,inner-1
    sum = sum + j
  end do
end subroutine innersub10000

subroutine innersub100000 (inner, sum)
  implicit none
  integer, intent(in) :: inner
  double precision, intent(inout) :: sum
  integer :: j
  do j=0,inner-1
    sum = sum + j
  end do
end subroutine innersub100000

subroutine innersub1000000 (inner, sum)
  implicit none
  integer, intent(in) :: inner
  double precision, intent(inout) :: sum
  integer :: j
  do j=0,inner-1
    sum = sum + j
  end do
end subroutine innersub1000000

subroutine innersub10000000 (inner, sum)
  implicit none
  integer, intent(in) :: inner
  double precision, intent(inout) :: sum
  integer :: j
  do j=0,inner-1
    sum = sum + j
  end do
end subroutine innersub10000000
