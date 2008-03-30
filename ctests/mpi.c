#include <unistd.h>
#include <stdio.h>
#include "../gptl.h"

#ifdef THREADED_OMP
#include <omp.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

int main (int argc, char **argv)
{
  int ret;
  int iam;
  int nompiter = 1;
  int i;

#ifndef HAVE_MPI
  printf ("HAVE_MPI is false so this test does nothing\n");
  return 0;
#endif

#ifdef THREADED_OMP
  nompiter = omp_get_max_threads ();
#endif

  ret = MPI_Init (&argc, &argv);
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);

  ret = GPTLsetoption (GPTLabort_on_error, 1);
  ret = GPTLinitialize ();

  /* Sleep for iam+mythread seconds */

#pragma omp parallel for private (i, ret)
  for (i = 0; i < nompiter; i++) {
    ret = GPTLstart ("main");
    ret = sleep (iam+i);
    ret = GPTLstop ("main");
  }

  ret = GPTLpr (iam);
  ret = MPI_Finalize ();

  return 0;
}
