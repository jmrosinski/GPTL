#include <stdio.h>
#include <unistd.h>  /* getopt */
#include <stdlib.h>  /* atof */

#ifdef HAVE_PAPI
#include <papi.h>
#endif

#if ( defined HAVE_LIBMPI ) || ( defined HAVE_LIBMPICH )
#include <mpi.h>
#endif

#include "../gptl.h"

int main (int argc, char **argv)
{
#if ( defined HAVE_LIBMPI ) || ( defined HAVE_LIBMPICH )
  extern char *optarg;
  void getmaxmin (const double *, int, double *, double *, int *, int *);

  const int mega = 1024*1024; /* convert to mega whatever */
  int sendtag;                /* tag for mpi */
  int recvtag;                /* tag for mpi */
  const double tol = 1.e-12;  /* tolerance (double precision) */

  int nsendrecv = 1024;       /* default number of doubles for send/receive */
  int niter = 1024;           /* default number of repetitions */
  int iam, sendto, recvfm;    /* mpi taskids */
  int ntask;                  /* number of mpi tasks */
  int c;                      /* for parsing argv */
  int i, iter;                /* loop indices */
  int nexpect;                /* expected return from PAPI */
  int onflg;                  /* returned by GPTLquery */
  int count;                  /* returned by GPTLquery */
  int taskmax, taskmin;       /* which mpi task caused a max or min */
  int ret;                    /* return code */

  long bytes;                 /* number of bytes */
  long papicounters[2];       /* returned from PAPI via GPTL */

  double *sendbuf;            /* send buffer for MPI_Sendrecv */
  double *recvbuf;            /* recv buffer for MPI_Sendrecv */
  double diff;                /* difference */
  double maxdiff = 0.;        /* max difference */
  double aggrate;             /* aggregate rate (summed over mpi tasks) */
  double rate, *rates;        /* rate for 1 task, rates for all tasks */
  double rmax, rmin;          /* max, min rates */
  double wmax, wmin;          /* wallclock max, min */
  double expect;              /* expected value */
  double wall, *walls;        /* wallclock time for 1 task, all tasks */
  double usr;                 /* usr CPU time returned by GPTLquery */
  double sys;                 /* sys CPU time returned by GPTLquery */
  double rdiff;               /* relative difference */
  double diam, drecvfm, diter;    /* double prec. versions of integers */
  double diampiter, drecvfmpiter; /* double prec. versions of integers */

  MPI_Status status;     /* required by MPI_Sendrecv */

  MPI_Init (&argc, &argv);
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);
  printf ("iam=%d\n", iam);
  ret = MPI_Comm_size (MPI_COMM_WORLD, &ntask);
  printf ("ntask=%d\n", ntask);

  walls = (double *) malloc (ntask * sizeof (double));
  rates = (double *) malloc (ntask * sizeof (double));
  
/* 
** Parse arg list
*/
  
  if (iam == 0) {
    while ((c = getopt (argc, argv, "l:n:")) != -1) {
      switch (c) {
      case 'l':
	nsendrecv = atoi (optarg);
	break;
      case 'n':
	niter = atoi (optarg);
	break;
      default:
	printf ("unknown option %c\n", c);
	exit (2);
      }
    }
  }

  printf ("Number of iterations  will be %d\n", nsendrecv);
  printf ("Loop length will be          %d\n", nsendrecv);

  sendbuf = (double *) malloc (nsendrecv * sizeof (double));
  recvbuf = (double *) malloc (nsendrecv * sizeof (double));

  /*
  ** Count floating point ops
  */

#ifdef HAVE_PAPI
  GPTLsetoption (PAPI_FP_OPS, 1);
#endif

  /*
  ** Try some different underlying timing routines
  */

  GPTLsetutr (GPTLclockgettime);
  GPTLsetutr (GPTLmpiwtime);
  GPTLsetutr (GPTLnanotime);
  GPTLsetutr (GPTLpapitime);

  if (GPTLinitialize () < 0)
    exit (1);

  sendto = (iam + 1) % ntask;         /* right neighbor */
  recvfm = (iam + ntask - 1) % ntask; /* left neighbor */

  diam = iam;
  drecvfm = recvfm;
  
  /*
  ** Loop over time doing the same thing over and over
  */

  for (iter = 0; iter < niter; iter++) {
    ret = MPI_Barrier (MPI_COMM_WORLD);

    diter = iter;
    diampiter = diam + diter;
    drecvfmpiter = drecvfm + diter;
    sendtag = iter;
    recvtag = iter;

    ret = GPTLstart ("FPops");
    
    /*
    ** Fill the send buffer with known values
    */

    for (i = 0; i < nsendrecv; i++) 
      sendbuf[i] = 0.1*(diampiter + (double) i);
    ret = GPTLstop ("FPops");

    /*
    ** Synchronize, then Sendrecv
    */

    ret = MPI_Barrier (MPI_COMM_WORLD);

    ret = GPTLstart ("sendrecv");
    ret = MPI_Sendrecv (sendbuf, nsendrecv, MPI_DOUBLE, sendto, sendtag,
			recvbuf, nsendrecv, MPI_DOUBLE, recvfm, recvtag,
			MPI_COMM_WORLD, &status);
    ret = GPTLstop ("sendrecv");
    
    ret = GPTLstart ("FPops");
    for (i = 0; i < nsendrecv; i++) {
      expect = 0.1*(drecvfmpiter + (double) i);

      /*
      ** On some machines the following is 3 FP ops
      */

      diff = abs ((expect - recvbuf[i]));

      /*
      ** On some machines the following is 2 FP ops
      */

      if (diff > maxdiff)
	maxdiff = diff;
    }
    ret = GPTLstop ("FPops");

    if (maxdiff / expect > tol) {
      printf ("iter %d task %d diff %f exceeds tolerance=%f\n", iter, iam, diff, tol);
      exit (1);
    }

    ret = MPI_Barrier (MPI_COMM_WORLD);

    ret = GPTLstart ("memBW");
    for (i = 0; i < nsendrecv; i++) {
      sendbuf[i] = 0.;
      recvbuf[i] = 0.;
    }
    ret = GPTLstop ("memBW");
  }

  /*
  ** Check results for accuracy and speed
  ** FP ops
  */

  printf ("FPops: iam %d maxdiff=%9.3g\n", iam, maxdiff);

  ret = GPTLquery ("FPops", -1, &count, &onflg, &wall, &usr, &sys, papicounters, 2);
  nexpect = niter*nsendrecv*(3 + 8);

#ifdef HAVE_PAPI
  rdiff = 100. * (nexpect - papicounters[0]) / (double) nexpect;
  printf ("FPops: iam %d expected %ld flops got %ld rdiff=%.2f%%\n",
	  iam, nexpect, papicounters[0], rdiff);
#endif
  
  ret = MPI_Gather (&wall, 1, MPI_DOUBLE, walls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (iam == 0) {
    getmaxmin (walls, ntask, &wmax, &wmin, &taskmax, &taskmin);
    rmax = nexpect / (mega*wmin);
    rmin = nexpect / (mega*wmax);
    aggrate = rmin * ntask; 
    printf ("FPops (Mflop/s task): max    =%9.3g %d\n"
	    "                      min    =%9.3g %d\n"
	    "                      aggrate=%9.3g\n",
	    rmax, taskmax, rmin, taskmin, aggrate);
  }
  
  /*
  ** MPI
  */

  ret = GPTLquery ("sendrecv", -1, &count, &onflg, &wall, &usr, &sys, papicounters, 0);
  bytes = sizeof (double) * niter * nsendrecv;
  rate =  bytes / (wall*mega);
  ret = MPI_Gather (&rate, 1, MPI_DOUBLE, rates, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  ret = MPI_Gather (&wall, 1, MPI_DOUBLE, walls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (iam == 0) {
    getmaxmin (walls, ntask, &wmax, &wmin, &taskmax, &taskmin);
    getmaxmin (rates, ntask, &rmax, &rmin, &taskmax, &taskmin);
    aggrate = (bytes * ntask) / (wmax*mega);
    printf ("sendrecv (MB/s task): max    =%9.3g %d\n"
	    "                      min    =%9.3g %d\n"
	    "                      aggrate=%9.3g\n",
	    rmax, taskmax, rmin, taskmin, aggrate);
  }

  /*
  ** Memory BW
  */

  ret = GPTLquery ("memBW", -1, &count, &onflg, &wall, &usr, &sys, papicounters, 0);
  bytes = sizeof (double) * niter * nsendrecv * 2;
  rate =  bytes / (wall*mega);
  ret = MPI_Gather (&rate, 1, MPI_DOUBLE, rates, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  ret = MPI_Gather (&wall, 1, MPI_DOUBLE, walls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (iam == 0) {
    getmaxmin (walls, ntask, &wmax, &wmin, &taskmax, &taskmin);
    getmaxmin (rates, ntask, &rmax, &rmin, &taskmax, &taskmin);
    aggrate = (bytes * ntask) / (mega*wmax);
    printf ("Mem BW   (MB/s task): max    =%9.3g %d\n"
	    "                      min    =%9.3g %d\n"
	    "                      aggrate=%9.3g\n",
	    rmax, taskmax, rmin, taskmin, aggrate);
  }

  GPTLpr (iam);
  MPI_Finalize ();
  return 0;
#else
  printf ("MPI not enabled so this code does nothing\n");
#endif
}

void getmaxmin (const double *vals, int ntask, double *vmax, double *vmin, int *indmax, int *indmin)
{
  int n;

  *vmax = vals[0];
  *vmin = vals[0];
  *indmax = 0;
  *indmin = 0;

  for (n = 1; n < ntask; n++) {
    if (vals[n] > *vmax) {
      *vmax = vals[n];
      *indmax = n;
    }
    if (vals[n] < *vmin) {
      *vmin = vals[n];
      *indmin = n;
    }
  }
}
