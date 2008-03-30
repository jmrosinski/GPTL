#include <unistd.h>
#include <stdio.h>
#include <papi.h>
#include "../gptl.h"

#ifdef THREADED_OMP
#include <omp.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

int main (int argc, char **argv)
{
  char c;

  int ret;
  int iam;
  int nompiter = 1;
  int i;
  int barriersync = 0;
  int commsize;

#ifndef HAVE_MPI
  printf ("HAVE_MPI is false so this test does nothing\n");
  return 0;
#endif

  while ((c = getopt (argc, argv, "b")) != -1) {
    switch (c) {
	case 'b':
	  barriersync = 1;
	  break;
	default:
	  printf ("unknown option %c\n", c);
	  return 1;
    }
  }
  
#ifdef THREADED_OMP
  nompiter = omp_get_max_threads ();
#endif

  ret = MPI_Init (&argc, &argv);
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);
  ret = MPI_Comm_size (MPI_COMM_WORLD, &commsize);

  if (iam == 0) {
    if (barriersync)
      printf ("barriersync is TRUE\n");
    else
      printf ("barriersync is FALSE\n");
  }

  ret = GPTLsetoption (GPTLoverhead, 0);
  ret = GPTLsetoption (GPTLabort_on_error, 1);
  ret = GPTLsetoption (PAPI_TOT_INS, 1);
  ret = GPTLinitialize ();

  ret = GPTLstart ("total");

  /* Sleep for iam+mythread seconds */

#pragma omp parallel for private (i, ret)
  for (i = 0; i < nompiter; i++) {
    ret = GPTLstart ("sleep(iam+mythread)");
    ret = sleep (iam+i);
    ret = GPTLstop ("sleep(iam+mythread)");
  }

  if (barriersync) {
    ret = GPTLstart ("barriersync");
    ret = MPI_Barrier (MPI_COMM_WORLD);
    ret = GPTLstop ("barriersync");
  }

  ret = GPTLstart ("sleep(1)");
  ret = MPI_Bcast (&ret, 1, MPI_INT, commsize - 1, MPI_COMM_WORLD);
  ret = sleep (1);
  ret = GPTLstop ("sleep(1)");

  ret = GPTLstop ("total");

  ret = GPTLpr (iam);
  ret = MPI_Finalize ();

  return 0;
}
