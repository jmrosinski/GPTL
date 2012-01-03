#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  /* getopt */
#include <string.h>  /* memset */

#include "../gptl.h"

#ifdef THREADED_OMP
#include <omp.h>
#endif

static int iam = 0;
static int nproc = 1;    /* number of MPI tasks (default 1) */
static int nthreads = 1; /* number of threads (default 1) */

double sub (int);

int main (int argc, char **argv)
{
  char pname[MPI_MAX_PROCESSOR_NAME];

  int iter;
  int counter;
  int c;
  int tnum = 0;
  int resultlen;

  double ret;

  extern char *optarg;

  while ((c = getopt (argc, argv, "p:")) != -1) {
    switch (c) {
    case 'p':
      if ((ret = GPTLevent_name_to_code (optarg, &counter)) != 0) {
	printf ("Failure from GPTLevent_name_to_code\n");
	return 1;
      }
      if (GPTLsetoption (counter, 1) < 0) {
	printf ("Failure from GPTLsetoption (%s,1)\n", optarg);
	return 1;
      }
      break;
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-p option_name]\n", argv[0]);
      return 2;
    }
  }
  
  (void) GPTLsetoption (GPTLabort_on_error, 1);
  (void) GPTLsetoption (GPTLoverhead, 1);
  (void) GPTLsetoption (GPTLnarrowprint, 1);

  if (MPI_Init (&argc, &argv) != MPI_SUCCESS) {
    printf ("Failure from MPI_Init\n");
    return 1;
  }

  /*
  ** If ENABLE_PMPI is set, GPTL was initialized in MPI_Init
  */

#ifndef ENABLE_PMPI
  (void) GPTLinitialize ();
  (void) GPTLstart ("total");
#endif
	 
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);
  ret = MPI_Comm_size (MPI_COMM_WORLD, &nproc);

  ret = MPI_Get_processor_name (pname, &resultlen);
  printf ("Rank %d is running on processor %s\n", iam, pname);

#ifdef THREADED_OMP
  nthreads = omp_get_max_threads ();
#pragma omp parallel for private (iter, ret, tnum)
#endif

  for (iter = 1; iter <= nthreads; iter++) {
#ifdef THREADED_OMP
    tnum = omp_get_thread_num ();
#endif
    printf ("Thread %d of rank %d on processor %s\n", tnum, iam, pname);
    ret = sub (iter);
  }

#ifndef ENABLE_PMPI
  (void) GPTLstop ("total");
  (void) GPTLpr (iam);
#endif

  if (iam == 0) {
    printf ("summary: testing GPTLpr_summary...\n");
    printf ("Number of threads was %d\n", nthreads);
    printf ("Number of tasks was %d\n", nproc);
  }

  if (GPTLpr_summary (MPI_COMM_WORLD) != 0) {
    return 1;
  }

  MPI_Finalize ();

  if (GPTLfinalize () != 0)
    return 1;

  return 0;
}

double sub (int iter)
{
  unsigned long usec;
  unsigned long looplen = iam*iter*100000;
  unsigned long i;
  double sum;

  (void) GPTLstart ("sub");
  /* Sleep msec is mpi rank + thread number */
  usec = 1000 * (iam * iter);

  (void) GPTLstart ("sleep");
  usleep (usec);
  (void) GPTLstop ("sleep");

  (void) GPTLstart ("work");
  sum = 0.;
  (void) GPTLstart ("add");
  for (i = 0; i < looplen; ++i) {
    sum += i;
  }
  (void) GPTLstop ("add");

  (void) GPTLstart ("madd");
  for (i = 0; i < looplen; ++i) {
    sum += i*1.1;
  }
  (void) GPTLstop ("madd");

  (void) GPTLstart ("div");
  for (i = 0; i < looplen; ++i) {
    sum /= 1.1;
  }
  (void) GPTLstop ("div");
  (void) GPTLstop ("work");
  (void) GPTLstop ("sub");
  return sum;
}
