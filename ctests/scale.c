#include <stdio.h>
#include <unistd.h>  /* getopt */
#include <stdlib.h>  /* atof */
#include <string.h>  /* atof */

#ifdef HAVE_PAPI
#include <papi.h>
#endif

#if ( defined HAVE_LIBMPI ) || ( defined HAVE_LIBMPICH )
#include <mpi.h>
#endif

#include "../gptl.h"

static int iam;                    /* my MPI rank */
static int ntask;                  /* number of mpi tasks */
static int nsendrecv = 1024000;    /* default number of doubles for send/receive */
static char procname[MPI_MAX_PROCESSOR_NAME];
static char **procnames;
static double totfp = 0.;          /* total floating point ops (est.) (init to 0) */
static double totmpi = 0.;         /* total items sent via MPI (init to 0) */
static double totmem = 0.;         /* total items copied in memory (init to 0) */

int main (int argc, char **argv)
{
#if ( defined HAVE_LIBMPI ) || ( defined HAVE_LIBMPICH )
  extern char *optarg;
  void getmaxmin (const double *, double *, double *, int *, int *);
  double chkresults (char *, int, int, double *);
  double fillsendbuf (double *, int);
  double zerobufs (double *, double *);
  void sendrecv (char *, double *, double *, int, int, int);
  void isendirecv (char *, double *, double *, int, int, int);
  void irecvisend (char *, double *, double *, int, int, int);
  int are_othernodes (int, int);
  int are_mynode (int, int);
  void gather_results (char *);

  int ppnmax = 6;             /* max # procs per node */
  int niter = 10;             /* default number of repetitions */

  int baseproc;
  int sendto;
  int recvfm;
  int c;                      /* for parsing argv */
  int iter;                   /* loop index */
  int t;                      /* task index */
  int ret;                    /* return code */
  int resultlen;              /* length of processor name */
  int ppnloc;                 /* # procs on this node */
  int nnode;                  /* # nodes */
  int code;

  double *sendbuf;            /* send buffer for MPI_Sendrecv */
  double *recvbuf;            /* recv buffer for MPI_Sendrecv */

  MPI_Status status;          /* required by MPI_Sendrecv */

  setlinebuf (stdout);
  MPI_Init (&argc, &argv);
  ret = MPI_Comm_rank (MPI_COMM_WORLD, &iam);
  ret = MPI_Get_processor_name (procname, &resultlen);
  ret = MPI_Comm_size (MPI_COMM_WORLD, &ntask);
  procnames = (char **) malloc (ntask * sizeof (char *));
  if (iam == 0) 
    printf ("ntask=%d\n", ntask);
  for (t = 0; t < ntask; t++) {
    procnames[t] = (char *) malloc (MPI_MAX_PROCESSOR_NAME+1);
  }
  if (iam == 0) {
    printf ("MPI task %d is running on %s\n", 0, procname);
    strcpy (procnames[0], procname);
    for (t = 1; t < ntask; t++) {
      MPI_Recv (procnames[t], MPI_MAX_PROCESSOR_NAME+1, MPI_CHAR, t, 0, MPI_COMM_WORLD, &status);
      printf ("MPI task %d is running on %s\n", t, procnames[t]);
    }
  } else {
    MPI_Send (procname, MPI_MAX_PROCESSOR_NAME+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  for (t = 0; t < ntask; t++) {
    MPI_Bcast (procnames[t], MPI_MAX_PROCESSOR_NAME+1, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  /*
  ** Determine ppnloc (needed by MPI in-memory section)
  ** Need to account for nnode not evenly dividing into ntask 
  */

  nnode = (ntask - 1)/ppnmax + 1;
  ppnloc = ppnmax;
  if (iam >= (nnode-1)*ppnmax)
    ppnloc = ntask - (nnode-1)*ppnmax;

  ret = MPI_Barrier (MPI_COMM_WORLD);

/* 
** Parse arg list
*/
  
  while ((c = getopt (argc, argv, "l:n:p:")) != -1) {
    switch (c) {
    case 'l':
      nsendrecv = atoi (optarg);
      break;
    case 'n':
      niter = atoi (optarg);
      break;
    case 'p':
      ppnmax = atoi (optarg);
      break;
    default:
      printf ("unknown option %c\n", c);
      exit (2);
    }
  }

  if (iam == 0) {
    printf ("Number of iterations  will be %d\n", niter);
    printf ("Loop length will be           %d\n", nsendrecv);
  }
  
  sendbuf = (double *) malloc (nsendrecv * sizeof (double));
  recvbuf = (double *) malloc (nsendrecv * sizeof (double));

  /*
  ** Count floating point ops
  */

  GPTLsetoption (GPTLverbose, 0);
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

  /*
  ** Loop over time doing the same thing over and over
  */

  for (iter = 0; iter < niter; iter++) {

    /*
    ** Base: one big ring from 0-ntask regardless of process to node mapping
    */

    sendto = (iam + 1) % ntask;         /* right neighbor */
    recvfm = (iam + ntask - 1) % ntask; /* left neighbor */
    sendrecv ("Sendrecv_base", sendbuf, recvbuf, iter, sendto, recvfm);
    isendirecv ("IsendIrecv_base", sendbuf, recvbuf, iter, sendto, recvfm); 
    irecvisend ("IrecvIsend_base", sendbuf, recvbuf, iter, sendto, recvfm); 

    /*
    ** Fabric: always send/recv to/from another node
    ** This can only be done when at least 2 nodes are filled with tasks
    */

    if (ntask >= 2*ppnmax) {
      sendto = (iam + ppnmax) % ntask;
      recvfm = (iam + ntask - ppnmax) % ntask;
      if (are_othernodes (sendto, recvfm)) {
	sendrecv ("Sendrecv_fabric", sendbuf, recvbuf, iter, sendto, recvfm);
	isendirecv ("IsendIrecv_fabric", sendbuf, recvbuf, iter, sendto, recvfm); 
	irecvisend ("IrecvIsend_fabric", sendbuf, recvbuf, iter, sendto, recvfm); 
      } else {
	printf ("iam=%d unexpectedly got same node for sendto=%d recvfm=%d\n",
		iam, sendto, recvfm);
	ret = MPI_Abort (MPI_COMM_WORLD, code);
      }
    }

    /*
    ** Memory: always send/recv to/from the same node
    */

    baseproc = (iam / ppnmax) * ppnmax;
    sendto = baseproc + (iam + 1) % ppnloc;
    recvfm = baseproc + (iam + ppnloc - 1) % ppnloc;
    if (are_mynode (sendto, recvfm)) {
      sendrecv ("Sendrecv_memory", sendbuf, recvbuf, iter, sendto, recvfm);
      isendirecv ("IsendIrecv_memory", sendbuf, recvbuf, iter, sendto, recvfm); 
      irecvisend ("IrecvIsend_memory", sendbuf, recvbuf, iter, sendto, recvfm); 
    } else {
      printf ("iam=%d unexpectedly got differet node for sendto=%d recvfm=%d\n",
	      iam, sendto, recvfm);
      ret = MPI_Abort (MPI_COMM_WORLD, code);
    }
  }

  gather_results ("FPOPS");
  gather_results ("MEMBW");

  gather_results ("Sendrecv_base");
  gather_results ("IsendIrecv_base");
  gather_results ("IrecvIsend_base");

  gather_results ("Sendrecv_fabric");
  gather_results ("IsendIrecv_fabric");
  gather_results ("IrecvIsend_fabric");

  gather_results ("Sendrecv_memory");
  gather_results ("IsendIrecv_memory");
  gather_results ("IrecvIsend_memory");

  GPTLpr (iam);
  MPI_Finalize ();
  return 0;
#else
  printf ("MPI not enabled so this code does nothing\n");
#endif
}

void getmaxmin (const double *vals, 
		double *vmax, 
		double *vmin, 
		int *indmax, 
		int *indmin)
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

double chkresults (char *label, 
		   int iter, 
		   int recvfm, 
		   double *recvbuf)
{
  int ifirst;
  int first;                  /* logical */
  int isave;
  int ret;
  int i;
  int code;

  double firstdiff;
  double expect;              /* expected value */
  double drecvfmpiter = recvfm + iter;
  double diff;                /* difference */
  double maxdiff = 0.;        /* max difference */
  double expectsave;

  const double tol = 1.e-12;  /* tolerance (double precision) */

  first = 1;

  ret = GPTLstart ("FPOPS");
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
      expectsave = expect;
    }
  }
  ret = GPTLstop ("FPOPS");
  
  if (maxdiff / expect > tol) {
    if (label)
      printf ("Problem in test %s\n", label);
    printf ("iter %d task %d worst diff expected %f got %f at i=%d\n", 
	    iter, iam, expectsave, recvbuf[isave], isave);
    printf ("    First diff exceeding tolerance is %f at i=%d\n", firstdiff, ifirst);
    ret = MPI_Abort (MPI_COMM_WORLD, code);
    exit (1);
  }
  
  /*
  ** Check results for accuracy and speed
  ** FP ops
  */

  if (maxdiff > 0.) 
    printf ("FPOPS: iam %d maxdiff=%9.3g\n", iam, maxdiff);

  return nsendrecv * 8;
}

double fillsendbuf (double *sendbuf, int iter)
{
  int ret;
  int i;

  const double diampiter = iam + iter;

  /*
  ** This should be 3 FP ops times nsendrecv
  */

  ret = GPTLstart ("FPOPS");
  for (i = 0; i < nsendrecv; i++) 
    sendbuf[i] = 0.1*(diampiter + (double) i);
  ret = GPTLstop ("FPOPS");

  return nsendrecv * 3;
}

double zerobufs (double *sendbuf, double *recvbuf)
{
  int ret;
  int i;

  /*
  ** This should be 2 times nsendrecv memory xfers
  */

  ret = GPTLstart ("MEMBW");
  for (i = 0; i < nsendrecv; i++) {
    sendbuf[i] = 0.;
    recvbuf[i] = 0.;
  }
  ret = GPTLstop ("MEMBW");
  return nsendrecv * 2;
}

void sendrecv (char *label, 
	       double *sendbuf, 
	       double *recvbuf, 
	       int iter, 
	       int sendto, 
	       int recvfm)
{
  int ret;
  const int sendtag = iter;
  const int recvtag = iter;

  MPI_Status status;          /* required by MPI_Sendrecv */

  ret = MPI_Barrier (MPI_COMM_WORLD);

  /*
  ** Fill send buffer, synchronize, Sendrecv, check results
  */

  totmem += zerobufs (sendbuf, recvbuf);         /* this accumulates into MEMBW */
  totfp += fillsendbuf (sendbuf, iter);     /* this accumulates into FPOPS */
  ret = MPI_Barrier (MPI_COMM_WORLD);

  ret = GPTLstart (label);
  ret = MPI_Sendrecv (sendbuf, nsendrecv, MPI_DOUBLE, sendto, sendtag,
		      recvbuf, nsendrecv, MPI_DOUBLE, recvfm, recvtag,
		      MPI_COMM_WORLD, &status);
  ret = GPTLstop (label);
  totfp += chkresults (label, iter, recvfm, recvbuf); /* this accumulates into FPOPS */
  totmpi += nsendrecv;                           /* number of doubles sent */
}

void isendirecv (char *label, 
		 double *sendbuf, 
		 double *recvbuf, 
		 int iter, 
		 int sendto, 
		 int recvfm)
{
  int ret;
  const int sendtag = iter;
  const int recvtag = iter;

  MPI_Request sendrequest;    /* required by MPI_Isend and Irecv */
  MPI_Request recvrequest;    /* required by MPI_Isend and Irecv */
  MPI_Status status;          /* required by MPI_Sendrecv */

  ret = MPI_Barrier (MPI_COMM_WORLD);

  /*
  ** Fill send buffer, synchronize, Isend_Irecv, check results
  */

  totmem += zerobufs (sendbuf, recvbuf);         /* this accumulates into MEMBW */
  totfp += fillsendbuf (sendbuf, iter);     /* this accumulates into FPOPS */
  ret = MPI_Barrier (MPI_COMM_WORLD);

  ret = GPTLstart (label);
  ret = MPI_Isend (sendbuf, nsendrecv, MPI_DOUBLE, sendto, sendtag,
		   MPI_COMM_WORLD, &sendrequest);
  ret = MPI_Irecv (recvbuf, nsendrecv, MPI_DOUBLE, recvfm, recvtag,
		   MPI_COMM_WORLD, &recvrequest);
  ret = MPI_Wait (&recvrequest, &status);
  ret = MPI_Wait (&sendrequest, &status);
  ret = GPTLstop (label);
  totfp += chkresults (label, iter, recvfm, recvbuf); /* this accumulates into FPOPS */
  totmpi += nsendrecv;                           /* number of doubles sent */
}

void irecvisend (char *label, 
		 double *sendbuf, 
		 double *recvbuf, 
		 int iter, 
		 int sendto, 
		 int recvfm)
{
  int ret;
  const int sendtag = iter;
  const int recvtag = iter;

  MPI_Request sendrequest;    /* required by MPI_Isend and Irecv */
  MPI_Request recvrequest;    /* required by MPI_Isend and Irecv */
  MPI_Status status;          /* required by MPI_Sendrecv */

  ret = MPI_Barrier (MPI_COMM_WORLD);

  /*
  ** Fill send buffer, synchronize, Isend_Irecv, check results
  */

  totmem += zerobufs (sendbuf, recvbuf);         /* this accumulates into MEMBW */
  totfp += fillsendbuf (sendbuf, iter);     /* this accumulates into FPOPS */
  ret = MPI_Barrier (MPI_COMM_WORLD);

  ret = GPTLstart (label);
  ret = MPI_Irecv (recvbuf, nsendrecv, MPI_DOUBLE, recvfm, recvtag,
		   MPI_COMM_WORLD, &recvrequest);
  ret = MPI_Isend (sendbuf, nsendrecv, MPI_DOUBLE, sendto, sendtag,
		   MPI_COMM_WORLD, &sendrequest);
  ret = MPI_Wait (&recvrequest, &status);
  ret = MPI_Wait (&sendrequest, &status);
  ret = GPTLstop (label);
  totfp += chkresults (label, iter, recvfm, recvbuf); /* this accumulates into FPOPS */
  totmpi += nsendrecv;                           /* number of doubles sent */
}

int are_othernodes (int sendto, int recvfm)
{
  return strcmp (procname, procnames[sendto]) != 0 && 
         strcmp (procname, procnames[recvfm]) != 0;
}

int are_mynode (int sendto, int recvfm)
{
  return strcmp (procname, procnames[sendto]) == 0 && 
         strcmp (procname, procnames[recvfm]) == 0;
}

void gather_results (char *label) 
{
  const int mega = 1024*1024; /* convert to mega whatever */
  int isfp = (strcmp (label, "FPOPS") == 0);
  int ret;
  int onflg;                  /* returned by GPTLquery */
  int count;                  /* returned by GPTLquery */
  int taskmax, taskmin;       /* which mpi task caused a max or min */

  long long papicounters[2];  /* returned from PAPI via GPTL */

  double rmax, rmin;          /* max, min rates */
  double wmax, wmin;          /* wallclock max, min */
  double wall, *walls;        /* wallclock time for 1 task, all tasks */
  double usr;                 /* usr CPU time returned by GPTLquery */
  double sys;                 /* sys CPU time returned by GPTLquery */
  double bytes;               /* number of bytes */
  double rdiff;               /* relative difference */
  double aggratemax;          /* aggregate rate (summed over mpi tasks) */
  double aggratemin;          /* aggregate rate (summed over mpi tasks) */

  char outfile[32];
  FILE *filep;                /* file pointer for GNUplot */

  walls = (double *) malloc (ntask * sizeof (double));
  
  /*
  ** FPOPS calcs.
  */
  
  if (GPTLquery (label, -1, &count, &onflg, &wall, &usr, &sys, papicounters, 2) == 0) {
    ret = MPI_Gather (&wall, 1, MPI_DOUBLE, walls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef HAVE_PAPI
    if (isfp) {
      rdiff = 100. * (totfp - papicounters[0]) / totfp;
      printf ("FPOPS: iam %d expected %g flops got %g rdiff=%9.2f\n",
	      iam, totfp, (double) papicounters[0], rdiff);
    }
#endif
    
    if (iam == 0) {
      getmaxmin (walls, &wmax, &wmin, &taskmax, &taskmin);
      if (wmin > 0.) {
	if (isfp) {
	  rmax = totfp / (mega * wmin);
	  rmin = totfp / (mega * wmax);
	} else {
	  bytes = sizeof (double) * totmpi;
	  rmax = bytes / (wmin*mega);
	  rmin = bytes / (wmax*mega);
	}
	
	aggratemax = rmax * ntask; 
	aggratemin = rmin * ntask; 
	
	if (isfp) {
	  printf ("FPOPS (Mflop/s task): max       =%9.3g %d\n"
		  "                      min       =%9.3g %d\n"
		  "                      aggratemax=%9.3g\n"
		  "                      aggratemin=%9.3g\n",
		  rmax, taskmax, rmin, taskmin, aggratemax, aggratemin);
	} else {
	  printf ("%20s (MB/s task): max       =%9.3g %d\n"
		  "                                  min       =%9.3g %d\n"
		  "                                  aggratemax=%9.3g\n"
		  "                                  aggratemin=%9.3g\n", label,
		  rmax, taskmax, rmin, taskmin, aggratemax, aggratemin);
	}
	strcpy (outfile, label);
	strcat (outfile, "_max");
	if ((filep = fopen (outfile, "a"))) {
	  fprintf (filep, "%d %9.3g\n", ntask, aggratemax);
	  (void) fclose (filep);
	}
	strcpy (outfile, label);
	strcat (outfile, "_min");
	if ((filep = fopen (outfile, "a"))) {
	  fprintf (filep, "%d %9.3g\n", ntask, aggratemin);
	  (void) fclose (filep);
	}
      }
    }
  }
  free (walls);
}
