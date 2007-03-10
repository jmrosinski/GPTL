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

static int iam, sendto, recvfm;    /* mpi taskids */
static int nsendrecv = 1024000;    /* default number of doubles for send/receive */
static double drecvfm;

int main (int argc, char **argv)
{
#if ( defined HAVE_LIBMPI ) || ( defined HAVE_LIBMPICH )
  FILE *filep;                /* file pointer for GNUplot */
  extern char *optarg;
  void getmaxmin (const double *, int, double *, double *, int *, int *);
  double chkresults (int, double *);
  double fillsendbuf (double *, double);
  double zerobufs (double *, double *);

  const int mega = 1024*1024; /* convert to mega whatever */
  int sendtag;                /* tag for mpi */
  int recvtag;                /* tag for mpi */

  int niter = 10;             /* default number of repetitions */
  int ntask;                  /* number of mpi tasks */
  int c;                      /* for parsing argv */
  int i, iter;                /* loop indices */
  int onflg;                  /* returned by GPTLquery */
  int count;                  /* returned by GPTLquery */
  int taskmax, taskmin;       /* which mpi task caused a max or min */
  int ret;                    /* return code */
  int resultlen;              /* length of processor name */

  long long papicounters[2];  /* returned from PAPI via GPTL */

  double *sendbuf;            /* send buffer for MPI_Sendrecv */
  double *recvbuf;            /* recv buffer for MPI_Sendrecv */
  double aggratemax;          /* aggregate rate (summed over mpi tasks) */
  double aggratemin;          /* aggregate rate (summed over mpi tasks) */
  double rmax, rmin;          /* max, min rates */
  double wmax, wmin;          /* wallclock max, min */
  double wall, *walls;        /* wallclock time for 1 task, all tasks */
  double usr;                 /* usr CPU time returned by GPTLquery */
  double sys;                 /* sys CPU time returned by GPTLquery */
  double rdiff;               /* relative difference */
  double diam, diter;         /* double prec. versions of integers */
  double diampiter, drecvfmpiter; /* double prec. versions of integers */
  double bytes;               /* number of bytes */
  double totfp;               /* total floating point ops (est.) */
  double totmpi;              /* total items sent via MPI */
  double totmem;              /* total items copied in memory */

  char procname[MPI_MAX_PROCESSOR_NAME];

  MPI_Status status;          /* required by MPI_Sendrecv */
  MPI_Request sendrequest;    /* required by MPI_Isend_Irecv */
  MPI_Request recvrequest;    /* required by MPI_Isend_Irecv */

  setlinebuf (stdout);
  MPI_Init (&argc, &argv);
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);
  ret = MPI_Get_processor_name (procname, &resultlen);
  printf ("MPI task %d is running on %s\n", iam, procname);
  ret = MPI_Comm_size (MPI_COMM_WORLD, &ntask);
  if (iam == 0)
    printf ("ntask=%d\n", ntask);
  ret = MPI_Barrier (MPI_COMM_WORLD);

  walls = (double *) malloc (ntask * sizeof (double));
  
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
    printf ("Number of iterations  will be %d\n", niter);
    printf ("Loop length will be           %d\n", nsendrecv);
  }
  
  /*
  ** All tasks need to know nsendrecv and niter
  */

  ret = MPI_Bcast (&nsendrecv, 1, MPI_INT, 0, MPI_COMM_WORLD);
  ret = MPI_Bcast (&niter, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
/*
  GPTLsetutr (GPTLclockgettime);
  GPTLsetutr (GPTLmpiwtime);
  GPTLsetutr (GPTLnanotime);
  GPTLsetutr (GPTLpapitime);
*/
  if (GPTLinitialize () < 0)
    exit (1);

  sendto = (iam + 1) % ntask;         /* right neighbor */
  recvfm = (iam + ntask - 1) % ntask; /* left neighbor */

  diam = iam;
  drecvfm = recvfm;
  
  /*
  ** Loop over time doing the same thing over and over
  */

  totfp = 0.;
  totmpi = 0.;
  totmem = 0.;

  for (iter = 0; iter < niter; iter++) {
    ret = MPI_Barrier (MPI_COMM_WORLD);

    diter = iter;
    diampiter = diam + diter;
    drecvfmpiter = drecvfm + diter;
    sendtag = iter;
    recvtag = iter;

    /*
    ** Fill send buffer, synchronize, Sendrecv, check results
    */

    totmem += zerobufs (sendbuf, recvbuf);         /* this accumulates into MemBW */
    totfp += fillsendbuf (sendbuf, diampiter);     /* this accumulates into FPops */
    ret = MPI_Barrier (MPI_COMM_WORLD);

    ret = GPTLstart ("MPI_Sendrecv");
    ret = MPI_Sendrecv (sendbuf, nsendrecv, MPI_DOUBLE, sendto, sendtag,
			recvbuf, nsendrecv, MPI_DOUBLE, recvfm, recvtag,
			MPI_COMM_WORLD, &status);
    ret = GPTLstop ("MPI_Sendrecv");
    totfp += chkresults (iter, recvbuf);           /* this accumulates into FPops */
    totmpi += nsendrecv;                           /* number of doubles sent */

    /*
    ** Fill send buffer, synchronize, Isend_Irecv, check results
    */

    totmem += zerobufs (sendbuf, recvbuf);
    totfp + fillsendbuf (sendbuf, diampiter);
    ret = MPI_Barrier (MPI_COMM_WORLD);

    ret = GPTLstart ("Isend_Irecv");
    ret = MPI_Isend (sendbuf, nsendrecv, MPI_DOUBLE, sendto, sendtag,
		     MPI_COMM_WORLD, &sendrequest);
    ret = MPI_Irecv (recvbuf, nsendrecv, MPI_DOUBLE, recvfm, recvtag,
		     MPI_COMM_WORLD, &recvrequest);
    ret = MPI_Wait (&recvrequest, &status);
    ret = MPI_Wait (&sendrequest, &status);
    ret = GPTLstop ("Isend_Irecv");
    totfp += chkresults (iter, recvbuf);           /* this accumulates into FPops */

    /*
    ** Fill send buffer, synchronize, Irecv_Isend, check results
    */

    totmem += zerobufs (sendbuf, recvbuf);
    totfp += fillsendbuf (sendbuf, diampiter);
    ret = MPI_Barrier (MPI_COMM_WORLD);

    ret = GPTLstart ("Irecv_Isend");
    ret = MPI_Irecv (recvbuf, nsendrecv, MPI_DOUBLE, recvfm, recvtag,
		     MPI_COMM_WORLD, &recvrequest);
    ret = MPI_Isend (sendbuf, nsendrecv, MPI_DOUBLE, sendto, sendtag,
		     MPI_COMM_WORLD, &sendrequest);
    ret = MPI_Wait (&recvrequest, &status);
    ret = MPI_Wait (&sendrequest, &status);
    ret = GPTLstop ("Irecv_Isend");
    totfp += chkresults (iter, recvbuf);           /* this accumulates into FPops */
  }

  /*
  ** FPops calcs.
  */

  ret = GPTLquery ("FPops", -1, &count, &onflg, &wall, &usr, &sys, papicounters, 2);
  ret = MPI_Gather (&wall, 1, MPI_DOUBLE, walls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef HAVE_PAPI
  rdiff = 100. * (totfp - papicounters[0]) / totfp;
  printf ("FPops: iam %d expected %g flops got %g rdiff=%9.2f\n",
	  iam, totfp, (double) papicounters[0], rdiff);
#endif

  if (iam == 0) {
    getmaxmin (walls, ntask, &wmax, &wmin, &taskmax, &taskmin);
    rmax = totfp / (mega * wmin);
    rmin = totfp / (mega * wmax);
    aggratemax = rmax * ntask; 
    aggratemin = rmin * ntask; 
    printf ("wmin=%9.3g rmax=%9.3g\n", wmin, rmax);
    printf ("FPops (Mflop/s task): max       =%9.3g %d\n"
	    "                      min       =%9.3g %d\n"
	    "                      aggratemax=%9.3g\n"
	    "                      aggratemin=%9.3g\n",
	    rmax, taskmax, rmin, taskmin, aggratemax, aggratemin);

    if (filep = fopen ("FPops_max", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemax);
      (void) fclose (filep);
    }
    if (filep = fopen ("FPops_min", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemin);
      (void) fclose (filep);
    }
  }
  
  /*
  ** MPI_Sendrecv
  */

  ret = GPTLquery ("MPI_Sendrecv", -1, &count, &onflg, &wall, &usr, &sys, papicounters, 0);
  ret = MPI_Gather (&wall, 1, MPI_DOUBLE, walls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (iam == 0) {
    getmaxmin (walls, ntask, &wmax, &wmin, &taskmax, &taskmin);
    bytes = sizeof (double) * totmpi;
    rmax = bytes / (wmin*mega);
    rmin = bytes / (wmax*mega);
    aggratemax = rmax * ntask;
    aggratemin = rmin * ntask;
    printf ("MPI_Sendrecv (MB/s task): max       =%9.3g %d\n"
	    "                          min       =%9.3g %d\n"
	    "                          aggratemax=%9.3g\n"
	    "                          aggratemin=%9.3g\n",
	    rmax, taskmax, rmin, taskmin, aggratemax, aggratemin);

    if (filep = fopen ("MPI_Sendrecv_max", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemax);
      (void) fclose (filep);
    }
    if (filep = fopen ("MPI_Sendrecv_min", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemin);
      (void) fclose (filep);
    }
  }

  /*
  ** MPI_Isend_Irecv
  */

  ret = GPTLquery ("Isend_Irecv", -1, &count, &onflg, &wall, &usr, &sys, papicounters, 0);
  ret = MPI_Gather (&wall, 1, MPI_DOUBLE, walls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (iam == 0) {
    getmaxmin (walls, ntask, &wmax, &wmin, &taskmax, &taskmin);
    bytes = sizeof (double) * totmpi;
    rmax = bytes / (wmin*mega);
    rmin = bytes / (wmax*mega);
    aggratemax = rmax * ntask;
    aggratemin = rmin * ntask;
    printf ("Isend_Irecv (MB/s task): max       =%9.3g %d\n"
	    "                         min       =%9.3g %d\n"
	    "                         aggratemax=%9.3g\n"
	    "                         aggratemin=%9.3g\n",
	    rmax, taskmax, rmin, taskmin, aggratemax, aggratemin);

    if (filep = fopen ("Isend_Irecv_max", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemax);
      (void) fclose (filep);
    }
    if (filep = fopen ("Isend_Irecv_min", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemin);
      (void) fclose (filep);
    }
  }

  /*
  ** MPI_Irecv_Isend
  */

  ret = GPTLquery ("Irecv_Isend", -1, &count, &onflg, &wall, &usr, &sys, papicounters, 0);
  ret = MPI_Gather (&wall, 1, MPI_DOUBLE, walls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (iam == 0) {
    getmaxmin (walls, ntask, &wmax, &wmin, &taskmax, &taskmin);
    bytes = sizeof (double) * totmpi;
    rmax = bytes / (wmin*mega);
    rmin = bytes / (wmax*mega);
    aggratemax = rmax * ntask;
    aggratemin = rmin * ntask;
    printf ("Irecv_Isend (MB/s task): max       =%9.3g %d\n"
	    "                         min       =%9.3g %d\n"
	    "                         aggratemax=%9.3g\n"
	    "                         aggratemin=%9.3g\n",
	    rmax, taskmax, rmin, taskmin, aggratemax, aggratemin);

    if (filep = fopen ("Irecv_Isend_max", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemax);
      (void) fclose (filep);
    }
    if (filep = fopen ("Irecv_Isend_min", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemin);
      (void) fclose (filep);
    }
  }

  /*
  ** Memory BW
  */

  ret = GPTLquery ("memBW", -1, &count, &onflg, &wall, &usr, &sys, papicounters, 0);
  ret = MPI_Gather (&wall, 1, MPI_DOUBLE, walls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (iam == 0) {
    getmaxmin (walls, ntask, &wmax, &wmin, &taskmax, &taskmin);
    bytes = sizeof (double) * totmem;
    rmax = bytes / (wmin*mega);
    rmin = bytes / (wmax*mega);
    aggratemax = rmax * ntask;
    aggratemin = rmin * ntask;
    printf ("Mem BW   (MB/s task): max       =%9.3g %d\n"
	    "                      min       =%9.3g %d\n"
	    "                      aggratemax=%9.3g\n"
	    "                      aggratemin=%9.3g\n",
	    rmax, taskmax, rmin, taskmin, aggratemax, aggratemin);

    if (filep = fopen ("MemBW_max", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemax);
      (void) fclose (filep);
    }
    if (filep = fopen ("MemBW_min", "a")) {
      fprintf (filep, "%d %9.3g\n", ntask, aggratemin);
      (void) fclose (filep);
    }
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

double chkresults (int iter, double *recvbuf)
{
  int ifirst;
  int first;                  /* logical */
  int isave;
  int ret;
  int i;

  double firstdiff;
  double expect;              /* expected value */
  double diter, drecvfmpiter;
  double diff;                /* difference */
  double maxdiff = 0.;        /* max difference */

  const double tol = 1.e-12;  /* tolerance (double precision) */

  first = 1;

  diter = iter;
  drecvfmpiter = drecvfm + diter;

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
    
    if (diff > maxdiff) {
      if (first) {
	firstdiff = diff;
	ifirst = i;
	first = 0;
      }
      maxdiff = diff;
      isave = i;
    }
  }
  ret = GPTLstop ("FPops");
  
  if (maxdiff / expect > tol) {
    printf ("iter %d task %d worst diff %f at i=%d exceeds tolerance=%f\n", 
	    iter, iam, diff, isave, tol);
    printf ("    First diff exceeding tolerance is %f at i=%d\n", firstdiff, ifirst);
    exit (1);
  }
  
  /*
  ** Check results for accuracy and speed
  ** FP ops
  */

  if (maxdiff > 0.) 
    printf ("FPops: iam %d maxdiff=%9.3g\n", iam, maxdiff);

  return nsendrecv * 8;
}

double fillsendbuf (double *sendbuf, double diampiter)
{
  int ret;
  int i;

  /*
  ** This should be 3 FP ops times nsendrecv
  */

  ret = GPTLstart ("FPops");
  for (i = 0; i < nsendrecv; i++) 
    sendbuf[i] = 0.1*(diampiter + (double) i);
  ret = GPTLstop ("FPops");

  return nsendrecv * 3;
}

double zerobufs (double *sendbuf, double *recvbuf)
{
  int ret;
  int i;

  /*
  ** This should be 2 times nsendrecv memory xfers
  */

  ret = GPTLstart ("memBW");
  for (i = 0; i < nsendrecv; i++) {
    sendbuf[i] = 0.;
    recvbuf[i] = 0.;
  }
  ret = GPTLstop ("memBW");
  return nsendrecv * 2;
}
