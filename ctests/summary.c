#include <stdio.h>
#include <stdlib.h>  /* atoi */
#include <unistd.h>  /* getopt */
#include <string.h>  /* memset */

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "../gptl.h"

#ifdef HAVE_PAPI
#include <papi.h>
#endif

static int iam = 0;
static int nproc = 1;    /* number of MPI tasks (default 1) */
static int nthreads = 1; /* number of threads (default 1) */

double sub (int);

int main (int argc, char **argv)
{
  int iter;
  int papiopt;
  int c;
  int comm = 0;

  double ret;

  extern char *optarg;

#ifdef HAVE_MPI
  if (MPI_Init (&argc, &argv) != MPI_SUCCESS) {
    printf ("Failure from MPI_Init\n");
    return 1;
  }
  comm = MPI_COMM_WORLD;
#endif

#ifdef HAVE_PAPI
  if (GPTL_PAPIlibraryinit () != 0) {
    printf ("Failure from GPTL_PAPIlibraryinit\n");
    return 1;
  }
#endif

  while ((c = getopt (argc, argv, "p:")) != -1) {
    switch (c) {
    case 'p':
#ifdef HAVE_PAPI
      if ((ret = PAPI_event_name_to_code (optarg, &papiopt)) != 0) {
	printf ("Failure from GPTL_PAPIname2id\n");
	return 1;
      }
      if (GPTLsetoption (papiopt, 1) < 0) {
	printf ("Failure from GPTLsetoption (%s,1)\n", optarg);
	return 1;
      }
#else
      printf ("HAVE_PAPI is false so -p ignored\n");
#endif
      break;
    default:
      printf ("unknown option %c\n", c);
      printf ("Usage: %s [-p papi_option_name]\n", argv[0]);
      return 2;
    }
  }
  
  (void) GPTLsetoption (GPTLabort_on_error, 1);
  (void) GPTLsetoption (GPTLoverhead, 1);
  (void) GPTLsetoption (GPTLnarrowprint, 1);

  (void) GPTLinitialize ();
  (void) GPTLstart ("total");
	 
#ifdef HAVE_MPI
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);
  ret = MPI_Comm_size (MPI_COMM_WORLD, &nproc);
#endif

#ifdef THREADED_OMP
  nthreads = omp_get_max_threads ();
#endif

#pragma omp parallel for private (iter, ret)

  for (iter = 1; iter <= nthreads; iter++) {
    ret = sub (iter);
  }

  (void) GPTLstop ("total");
  (void) GPTLpr (iam);

  if (iam == 0)
    printf ("summary: testing GPTLpr_summary...\n");

  if (GPTLpr_summary (comm) == 0) {
    if (iam == 0)
      printf ("Success\n");
  } else {
    if (iam == 0)
      printf ("Failure\n");
    return 1;
  }
  if (GPTLfinalize () != 0)
    return 1;

#ifdef HAVE_MPI
  MPI_Finalize ();
#endif
  return 0;
}

double sub (int iter)
{
  unsigned long usec;
  unsigned long looplen = iam*iter*10000;
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
