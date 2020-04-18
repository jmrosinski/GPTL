#include "config.h"
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  /* getopt */
#include <string.h>  /* memset */

#include "gptl.h"
#include "gptlmpi.h"

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

  int t;
  int counter;
  int c;
  int tnum = 0;
  int resultlen;
  int ret;
  double value;
  extern char *optarg;

  while ((c = getopt (argc, argv, "p:")) != -1) {
    switch (c) {
#ifdef HAVE_PAPI
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
 #endif
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-p option_name]\n", argv[0]);
      return 2;
    }
  }
  
  ret = GPTLsetoption (GPTLabort_on_error, 1);
  ret = GPTLsetoption (GPTLoverhead, 1);

  if (MPI_Init (&argc, &argv) != MPI_SUCCESS) {
    printf ("Failure from MPI_Init\n");
    return 1;
  }

  ret = GPTLinitialize ();
	 
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);
  ret = MPI_Comm_size (MPI_COMM_WORLD, &nproc);

  ret = MPI_Get_processor_name (pname, &resultlen);
  printf ("Rank %d is running on processor %s\n", iam, pname);

#ifdef THREADED_OMP
  nthreads = omp_get_max_threads ();
#pragma omp parallel for private (t, ret, tnum)
#endif

  for (t = 0; t < nthreads; ++t) {
#ifdef THREADED_OMP
    tnum = omp_get_thread_num ();
#endif
    printf ("Thread %d of rank %d on processor %s\n", tnum, iam, pname);
    // Have rank 1 not have any regions.
    if (iam != 1)
      value = sub (t);
  }

  ret = GPTLpr (iam);

  if (iam == 0) {
    printf ("summary: testing GPTLpr_summary...\n");
    printf ("Number of threads was %d\n", nthreads);
    printf ("Number of tasks was %d\n", nproc);
  }

  // NOTE: if ENABLE_PMPI is set, 2nd pr call below will show some extra send/recv calls
  // due to MPI calls from within GPTLpr_summary_file
  if (GPTLpr_summary (MPI_COMM_WORLD) != 0)
    return 1;

  if (GPTLpr_summary_file (MPI_COMM_WORLD, "timing.summary.duplicate") != 0)
    return 1;

  ret = MPI_Finalize ();

  if (GPTLfinalize () != 0)
    return 1;

  return 0;
}

double sub (int t)
{
  unsigned long usec;
  unsigned long looplen = iam*t*100000;
  unsigned long i;
  double sum;
  int ret;

  ret = GPTLstart ("sub");
  /* Sleep msec is mpi rank + thread number */
  usec = 1000 * (iam + t)*10;

  ret = GPTLstart ("sleep");
#ifdef THREADED_OMP
  printf ("iam %d thread %d sleeping %d ms\n", iam, omp_get_thread_num(), (int) usec/1000);
#else
  printf ("iam %d sleeping %d ms\n", iam, (int) usec/1000);
#endif
  usleep (usec);
  ret = GPTLstop ("sleep");

  ret = GPTLstart ("work");
  sum = 0.;
  ret = GPTLstart ("add");
  for (i = 0; i < looplen; ++i) {
    sum += i;
  }
  ret = GPTLstop ("add");

  ret = GPTLstart ("madd");
  for (i = 0; i < looplen; ++i) {
    sum += i*1.1;
  }
  ret = GPTLstop ("madd");

  ret = GPTLstart ("div");
  for (i = 0; i < looplen; ++i) {
    sum /= 1.1;
  }
  ret = GPTLstop ("div");
  ret = GPTLstop ("work");
  ret = GPTLstop ("sub");
  return sum;
}
