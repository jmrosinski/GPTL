#include <stdio.h>
#include <stdlib.h>

__device__ static bool scalar;
__device__ static int *arr;

extern "C" {
  __device__ int sub1 (bool arg1, int arg2)
{
  printf ("Entered sub1\n");
  scalar = arg1;
  arr = (int *) malloc (10 * sizeof (int));
  arr[0] = arg2;
  printf ("sub1: set scalar=%d\n",scalar);
  printf ("sub1: malloced arr at address=%p\n", arr);
  printf ("sub1: set arr[0]=%d\n", arr[0]);
  return 0;
}

__device__ int sub2 (const char *arg)
{
  printf ("sub2: arg=%s\n", arg);
  printf ("sub2: scalar=%d\n", scalar);
  printf ("sub2: address of arr=%p\n", arr);
  printf ("sub2: arr[0]=%d\n", arr[0]);
  return 0;
}

}
