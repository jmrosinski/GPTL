#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#ifdef HAVE_PAPI
#include <papi.h>
#endif
#include "../gptl.h"

int main (int argc, char **argv)
{
  int iam;
  int ret;

  ret = MPI_Init (&argc, &argv);
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);

  ret = GPTLsetoption (GPTLoverhead, 0);
#ifdef HAVE_PAPI
  ret = GPTLsetoption (PAPI_TOT_CYC, 1);
  ret = GPTLsetoption (PAPI_TOT_IIS, 1);
#endif

  if ((ret = GPTLinitialize ()) != 0) {
    printf ("load_imbalance: GPTLinitialize failure\n");
    return -1;
  }

  ret = GPTLstart ("total");
  ret = GPTLstart ("sleep_iam_A");
  ret = sleep (iam);
  ret = GPTLstop ("sleep_iam_A");

  ret = GPTLstart ("sleep_2_Barrier_A");
  ret = sleep (2);
  ret = MPI_Barrier (MPI_COMM_WORLD);
  ret = GPTLstop ("sleep_2_Barrier_A");

  ret = GPTLstart ("sleep_iam_B");
  ret = sleep (iam);
  ret = GPTLstop ("sleep_iam_B");

  ret = GPTLstart ("sync_B");
  ret = MPI_Barrier (MPI_COMM_WORLD);
  ret = GPTLstop ("sync_B");

  ret = GPTLstart ("sleep_2_B");
  ret = sleep (2);
  ret = GPTLstop ("sleep_2_B");
  ret = GPTLstop ("total");

  ret = GPTLpr (iam);

  ret = MPI_Finalize ();
  return 0;
}
