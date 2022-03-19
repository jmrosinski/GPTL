#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "gptl.h"
#include "gptlmpi.h"

const MPI_Comm comm = MPI_COMM_WORLD;
const int tag = 98;

int send_recv (int, int);
int ssend_recv (int, int);
int sendrecv (int, int);
int irecv_isend_wait (int, int);
int irecv_isend_waitall (int, int);
int issend_recv (int, int);
int bcast (int, int);
int reduce (int, int);
int allreduce (int, int);
int alltoall (int, int);
int alltoallv (int, int);
int gather (int, int);
int scatter (int, int);
int chkbuf (int, const int, const char *, const int, const int);

int main (int argc, char **argv)
{
  int iam;
  int i, ret;
  int commsize;
  int resultlen;                      /* returned length of string from MPI routine */
  int provided;                       /* level of threading support in this MPI lib */
  char string[MPI_MAX_ERROR_STRING];  /* character string returned from MPI routine */
  const char *mpiroutine[] = {"MPI_Ssend", "MPI_Send", "MPI_Recv", "MPI_Sendrecv", "MPI_Irecv",
			      "MPI_Isend", "MPI_Waitall", "MPI_Barrier", "MPI_Bcast",
			      "MPI_Allreduce", "MPI_Gather", "MPI_Scatter", "MPI_Alltoall",
			      "MPI_Reduce", "MPI_Issend", "MPI_Alltoallv"};
  const int nroutines = sizeof (mpiroutine) / sizeof (char *);
  double wallclock;

  /*
  int DebugWait = 1;
  while (DebugWait) {
  }
  */

  /* Initialize MPI by using MPI_Init_thread: report back level of MPI support */
  if ((ret = MPI_Init_thread (&argc, &argv, MPI_THREAD_SINGLE, &provided)) != 0) {
    MPI_Error_string (ret, string, &resultlen);
    printf ("%s: error from MPI_Init_thread: %s\n", argv[0], string);
    MPI_Abort (comm, -1);
  }
  
  ret = MPI_Comm_rank (comm, &iam);            /* Get my rank */
  ret = MPI_Comm_size (comm, &commsize);       /* Get communicator size */
  if (commsize % 2 != 0) {
    printf ("%s requires number of procs to be EVEN\n", argv[0]);
    MPI_Abort (comm, -1);
  }

  ret = GPTLsetoption (GPTLoverhead, 0);       /* Don't print overhead stats */
  ret = GPTLsetoption (GPTLpercent, 0);        /* Don't print percentage stats */
  ret = GPTLsetoption (GPTLabort_on_error, 1); /* Abort on any GPTL error */

  ret = GPTLinitialize ();                     /* Initialize GPTL */
  ret = GPTLstart ("total");                   /* Time the whole program */

  if (iam == 0) {
    printf ("%s: testing suite of MPI routines for auto-instrumentation via GPTL PMPI layer\n",
	    argv[0]);
    switch (provided) {
    case MPI_THREAD_SINGLE:
      printf ("MPI support level is MPI_THREAD_SINGLE\n");
      break;
    case MPI_THREAD_SERIALIZED:
      printf ("MPI support level is MPI_THREAD_SERIALIZED\n");
      break;
    case MPI_THREAD_MULTIPLE:
      printf ("MPI support level is MPI_THREAD_MULTIPLE\n");
      break;
    default:
      printf ("MPI support level is not known\n");
      MPI_Abort (comm, -1);
    }
  }

  if ((ret = MPI_Barrier (comm)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = send_recv (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = ssend_recv (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = sendrecv (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = irecv_isend_wait (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = irecv_isend_waitall (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = bcast (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = allreduce (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = issend_recv (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = alltoall (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = alltoallv (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = reduce (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = gather (iam, commsize)) != 0)
    MPI_Abort (comm, -1);
  if ((ret = scatter (iam, commsize)) != 0)
    MPI_Abort (comm, -1);

  // Change 0 to 1 to verify test script fails correctly
  if (0)
    MPI_Abort (comm, -1);
  
  ret = GPTLstop ("total");
  ret = GPTLpr (iam);             /* Print the results for my rank */
  ret = GPTLpr_summary (comm);    /* Print the results summary across ranks */
  
  /* Check that PMPI entries were generated for all expected routines */
  if (iam == 0) {
    for (i = 0; i < nroutines; ++i) {
      printf ("%s: checking that there is a GPTL entry for MPI routine %s...\n",
	      argv[0], mpiroutine[i]);
      ret = GPTLget_wallclock (mpiroutine[i], 0, &wallclock);
      if (ret < 0) {
	printf ("Failure\n");
	return -1;
      }
      printf("Success\n");
    }
  }
  ret = MPI_Finalize ();
  return 0;
}

int send_recv (int myid, int numprocs)
{
  int ret;
  int recvbuf[1] = {-1};
  int dest       = (myid+1) % numprocs;
  int sendbuf[1] = {dest};
  int source     = myid == 0 ? numprocs-1: myid-1;
  MPI_Status status;
  static const char *thisfunc = "send_recv";
  
  if (myid % 2 == 0) {
    ret = MPI_Send (sendbuf, 1, MPI_INT, dest, tag, comm);
    ret = MPI_Recv (recvbuf, 1, MPI_INT, source, tag, comm, &status);
  } else {
    ret = MPI_Recv (recvbuf, 1, MPI_INT, source, tag, comm, &status);
    ret = MPI_Send (sendbuf, 1, MPI_INT, dest, tag, comm);
  }
  return chkbuf (ret, myid, thisfunc, recvbuf[0], myid);
}
 
int ssend_recv (int myid, int numprocs)
{
  int ret;
  int recvbuf[1] = {-1};
  int dest       = (myid+1) % numprocs;
  int sendbuf[1] = {dest};
  int source     = myid == 0 ? numprocs-1: myid-1;
  MPI_Status status;
  static const char *thisfunc = "ssend_recv";
  
  if (myid % 2 == 0) {
    ret = MPI_Ssend (sendbuf, 1, MPI_INT, dest, tag, comm);
    ret = MPI_Recv  (recvbuf, 1, MPI_INT, source, tag, comm, &status);
  } else {
    ret = MPI_Recv (recvbuf, 1, MPI_INT, source, tag, comm, &status);
    ret = MPI_Ssend (sendbuf, 1, MPI_INT, dest, tag, comm);
  }
  return chkbuf (ret, myid, thisfunc, recvbuf[0], myid);
}

int sendrecv (int myid, int numprocs)
{
  int ret = 0;
  int source     = myid == 0 ? numprocs-1: myid-1;
  int dest       = (myid+1) % numprocs;
  int sendbuf[1] = {dest};
  int recvbuf[1] = {-1};
  MPI_Status status;
  static const char *thisfunc = "sendrecv";

  ret = MPI_Sendrecv (sendbuf, 1, MPI_INT, dest, tag, 
		      recvbuf, 1, MPI_INT, source, tag, 
		      comm, &status);
  return chkbuf (ret, myid, thisfunc, recvbuf[0], myid);
}

int irecv_isend_wait (int myid, int numprocs)
{
  int ret;
  int source     = myid == 0 ? numprocs-1: myid-1;
  int dest       = (myid+1) % numprocs;
  int sendbuf[1] = {dest};
  int recvbuf[1] = {-1};
  MPI_Request sendreq, recvreq;
  MPI_Status status;
  static const char *thisfunc = "isend_irecv_wait";

  ret = MPI_Irecv (recvbuf, 1, MPI_INT, source, tag, comm, &recvreq);
  ret = MPI_Isend (sendbuf, 1, MPI_INT, dest, tag, comm, &sendreq);
  ret = MPI_Wait (&recvreq, &status);
  ret = MPI_Wait (&sendreq, &status);
  return chkbuf (ret, myid, thisfunc, recvbuf[0], myid);
}
  
int irecv_isend_waitall (int myid, int numprocs)
{
  int ret;
  int source     = myid == 0 ? numprocs-1: myid-1;
  int dest       = (myid+1) % numprocs;
  int sendbuf[1] = {dest};
  int recvbuf[1] = {-1};
  MPI_Request sendreq, recvreq;
  MPI_Status status;
  static const char *thisfunc = "irecv_isend_waitall";

  ret = MPI_Irecv (recvbuf, 1, MPI_INT, source, tag, comm, &recvreq);
  ret = MPI_Isend (sendbuf, 1, MPI_INT, dest, tag, comm, &sendreq);
  ret = MPI_Waitall (1, &recvreq, &status);
  ret = MPI_Waitall (1, &sendreq, &status);
  return chkbuf (ret, myid, thisfunc, recvbuf[0], myid);
}

int issend_recv (int myid, int numprocs)
{
  int ret;
  int recvbuf[1] = {-1};
  int source     = myid == 0 ? numprocs-1: myid-1;
  int dest       = (myid+1) % numprocs;
  int sendbuf[1] = {dest};
  MPI_Request sendreq;
  MPI_Status status;
  static const char *thisfunc = "issend_recv";
  
  if (myid % 2 == 0) {
    ret = MPI_Issend (sendbuf, 1, MPI_INT, dest, tag, comm, &sendreq);
    ret = MPI_Recv  (recvbuf, 1, MPI_INT, source, tag, comm, &status);
  } else {
    ret = MPI_Recv (recvbuf, 1, MPI_INT, source, tag, comm, &status);
    ret = MPI_Issend (sendbuf, 1, MPI_INT, dest, tag, comm, &sendreq);
  }
  return chkbuf (ret, myid, thisfunc, recvbuf[0], myid);
}

int bcast (int myid, int numprocs)
{
  int ret;
  const int val = 7;
  int buf[1]    = {-1};
  static const char *thisfunc = "bcast";

  if (myid == 0)
    buf[0] = val;
  ret = MPI_Bcast (buf, 1, MPI_INT, 0, comm);
  return chkbuf (ret, myid, thisfunc, buf[0], val);
}

int reduce (int myid, int numprocs)
{
  int ret;
  int sendbuf[1] = {myid};
  int recvbuf[1] = {-1};
  static const char *thisfunc = "reduce";

  ret = MPI_Reduce (sendbuf, recvbuf, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  if (myid == 0)
    return chkbuf (ret, myid, thisfunc, recvbuf[0], numprocs-1);
  else
    return ret;
}
  
int allreduce (int myid, int numprocs)
{
  int ret;
  int sendbuf[1] = {myid};
  int recvbuf[1] = {-1};
  static const char *thisfunc = "allreduce";

  if (myid == 0)
    recvbuf[0] = myid;
  else
    sendbuf[0] = myid;
  
  ret = MPI_Allreduce (sendbuf, recvbuf, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return chkbuf (ret, myid, thisfunc, recvbuf[0], numprocs-1);
}
  
int gather (int myid, int numprocs)
{
  int n;
  int ret = 0;
  int sendbuf[1];
  int recvbuf[numprocs];
  static const char *thisfunc = "gather";

  for (n = 0; n < numprocs; ++n)
    recvbuf[n] = -1;

  sendbuf[0] = myid;
  ret = MPI_Gather (sendbuf, 1, MPI_INT,
		    recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (myid == 0) {
    for (n = 0; n < numprocs; ++n) {
      if (chkbuf (ret, myid, thisfunc, recvbuf[n], n) != 0)
	ret = -1;
    }
  }
  return ret;
}

int scatter (int myid, int numprocs)
{
  int n;
  int ret;
  int sendbuf[numprocs];
  int recvbuf[1] = {-1};
  static const char *thisfunc = "scatter";

  if (myid == 0) {
    for (n = 0; n < numprocs; ++n)
      sendbuf[n] = n;
  }
  ret = MPI_Scatter (sendbuf, 1, MPI_INT,
		     recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return chkbuf (ret, myid, thisfunc, recvbuf[0], myid);
}

int alltoall (int myid, int numprocs)
{
  int n;
  int ret = 0;
  int sbuf[numprocs];
  int rbuf[numprocs];
  static const char *thisfunc = "alltoall";

  for (n = 0; n < numprocs; ++n) {
    rbuf[n] = -1;
    sbuf[n] = n;
  }
  ret = MPI_Alltoall (sbuf, 1, MPI_INT,
		      rbuf, 1, MPI_INT, MPI_COMM_WORLD);
  for (n = 0; n < numprocs; ++n) {
    if (chkbuf (ret, myid, thisfunc, rbuf[n], myid) != 0)
      ret = -1;
  }
  return ret;
}

int alltoallv (int myid, int numprocs)
{
  int n;
  int ret = 0;
  int sbuf[numprocs];
  int scounts[numprocs];
  int sdispls[numprocs];
  int rbuf[numprocs];
  int rcounts[numprocs];
  int rdispls[numprocs];
  static const char *thisfunc = "alltoallv";

  for (n = 0; n < numprocs; ++n) {
    sbuf[n] = n;
    scounts[n] = 1;
    sdispls[n] = n;
    rbuf[n] = -1;
    rcounts[n] = 1;
    rdispls[n] = n;
  }
  
  ret = MPI_Alltoallv (sbuf, scounts, sdispls, MPI_INT,
		       rbuf, rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
  for (n = 0; n < numprocs; ++n) {
    if (chkbuf (ret, myid, thisfunc, rbuf[n], myid) != 0)
      ret = -1;
  }
  return ret;
}

int chkbuf (int ret, const int rank, const char *func, const int got, const int shouldbe)
{
  if (ret != 0) {
    printf ("%s rank %d: failure\n", func, rank);
    return ret;
  }
  if (got == shouldbe) {
    printf ("%s rank %d success\n", func, rank);
  } else {
    ret = -1;
    printf ("%s rank %d failure got %d should have got %d\n", func, rank, got, shouldbe);
  }
  return ret;
}
