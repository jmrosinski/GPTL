#include "config.h"
#include <stdio.h>
#include <unistd.h>  /* sleep, usleep */
#include "gptl.h"

#ifdef HAVE_LIBMPI
#include <mpi.h>
#endif

#ifdef THREADED_OMP
#include <omp.h>
#endif

int main (int argc, char **argv)
{
  int iam = 0;
  int nranks = 1;    /* number of MPI tasks (default 1) */
  int nthreads = 1;  /* number of threads (default 1) */
  int iter;
  int tnum = 0;
  MPI_Comm comm;
#ifdef HAVE_PAPI
  int code;
#endif
  int ret;
  useconds_t msec_sleep;  /* number of msec to sleep */

#ifdef HAVE_PAPI
  int sub (int, int);
#endif

  /* ret = GPTLsetoption (GPTLabort_on_error, 1); */
#ifdef HAVE_PAPI
  ret = GPTLevent_name_to_code ("PAPI_FP_OPS", &code);
  if (ret == 0) {
    printf ("Enabling option PAPI_FP_OPS\n");
    ret = GPTLsetoption (code, 1);
  } else {
    printf ("Unable to get option for PAPI_FP_OPS\n");
  }
#endif

#ifdef HAVE_LIBMPI
  if (MPI_Init (&argc, &argv) != MPI_SUCCESS) {
    printf ("Failure from MPI_Init\n");
    return 1;
  }
  comm = MPI_COMM_WORLD;
  ret = MPI_Comm_rank (comm, &iam);
  ret = MPI_Comm_size (comm, &nranks);
#endif

  ret = GPTLinitialize ();
  ret = GPTLstart ("total");
	 
#ifdef THREADED_OMP
  nthreads = omp_get_max_threads ();
#pragma omp parallel for private (ret, tnum, msec_sleep)
#endif
  for (iter = 0; iter < nthreads; ++iter) {
#ifdef THREADED_OMP
    tnum = omp_get_thread_num ();
#endif
    /* Test 1: threaded sleep */
    ret = GPTLstart ("nranks-iam+mythread");
    msec_sleep = (useconds_t) (1000*(nranks - iam + tnum));
    ret = usleep (msec_sleep);
    ret = GPTLstop ("nranks-iam+mythread");
  }

  /* Test 2: 5-task usleep(f(iam)) ms */
  if (iam > 0 && iam < 6) {
    ret = GPTLstart ("1-5_iam");
    msec_sleep = (useconds_t) (1000*iam);
    ret = usleep (msec_sleep);
    ret = GPTLstop ("1-5_iam");
  }

#ifdef HAVE_PAPI
  /* Test 3: PAPI */
  ret = GPTLstart ("1e3*iam*mythread_FP_OPS");
  ret = sub (iam, tnum);
  ret = GPTLstop ("1e3*iam*mythread_FP_OPS");
#endif

  ret = GPTLstop ("total");
  ret = GPTLpr (iam);

  if (iam == 0)
    printf ("global: testing GPTLpr_summary...\n");

  if (GPTLpr_summary (comm) != 0)
    return 1;
#ifdef HAVE_LIBMPI
  ret = MPI_Finalize ();
#endif

  if (GPTLfinalize () != 0)
    return 1;

  return 0;
}

#ifdef HAVE_PAPI
int sub (int iam, int tnum)
{
  float sum;
  int i;

  sum = 1.7;
  for (i = 0; i < iam*tnum; ++i)
    sum *= 0.999;
  printf ("sum=%f\n", sum);
  return 0;
}
#endif
