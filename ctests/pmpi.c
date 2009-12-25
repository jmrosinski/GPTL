#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "../gptl.h"

static const MPI_Comm comm = MPI_COMM_WORLD;
static int iam;

int main (int argc, char **argv)
{
  int i, ret;
  int commsize;
  int val;
  const int count = 1024;
  const int tag = 98;
  int sendbuf[count];
  int recvbuf[count];
  int *gsbuf;
  int *atoabufsend, *atoabufrecv;
  int sum;
  MPI_Status status;
  MPI_Request request;
  int dest;
  int source;

  void chkbuf (const char *, int *, const int, const int);

  //  int DebugWait = 1;

  //  while (DebugWait) {
  //  }

  ret = MPI_Init (&argc, &argv);               // Initialize MPI
  ret = MPI_Comm_rank (comm, &iam);            // Get my rank
  ret = MPI_Comm_size (comm, &commsize);       // Get communicator size

  ret = GPTLsetoption (GPTLoverhead, 0);       // Don't print overhead stats
  ret = GPTLsetoption (GPTLpercent, 0);        // Don't print percentage stats
  ret = GPTLsetoption (GPTLabort_on_error, 1); // Abort on any GPTL error

  ret = GPTLinitialize ();                     // Initialize GPTL
  ret = GPTLstart ("total");                   // Time the whole program

  for (i = 0; i < count; ++i)
    sendbuf[i] = iam;

  dest = (iam + 1)%commsize;
  source = iam - 1;
  if (source < 0)
    source = commsize - 1;

  if (commsize % 2 == 0) {
    if (iam % 2 == 0) {
      ret = MPI_Send (sendbuf, count, MPI_INT, dest, tag, comm);
      ret = MPI_Recv (recvbuf, count, MPI_INT, source, tag, comm, &status);
    } else {
      ret = MPI_Recv (recvbuf, count, MPI_INT, source, tag, comm, &status);
      ret = MPI_Send (sendbuf, count, MPI_INT, dest, tag, comm);
    }
  }
  chkbuf ("mpi_send + mpi_recv", recvbuf, count, source);

  ret = MPI_Sendrecv (sendbuf, count, MPI_INT, dest, tag, 
		      recvbuf, count, MPI_INT, source, tag, 
		      comm, &status);
  chkbuf ("MPI_Sendrecv", recvbuf, count, source);

  ret = MPI_Irecv (recvbuf, count, MPI_INT, source, tag, 
		   comm, &request);
  ret = MPI_Isend (sendbuf, count, MPI_INT, dest, tag, 
		   comm, &request);
  ret = MPI_Wait (&request, &status);
  chkbuf ("MPI_Wait", recvbuf, count, source);

  ret = MPI_Irecv (recvbuf, count, MPI_INT, source, tag, 
		   comm, &request);
  ret = MPI_Isend (sendbuf, count, MPI_INT, dest, tag, 
		   comm, &request);
  ret = MPI_Waitall (1, &request, &status);
  chkbuf ("MPI_Waitall", recvbuf, count, source);

  ret = MPI_Barrier (comm);

  ret = MPI_Bcast (sendbuf, count, MPI_INT, 0, comm);
  chkbuf ("MPI_Bcast", sendbuf, count, 0);

  for (i = 0; i < count; ++i)
    sendbuf[i] = iam;

  ret = MPI_Allreduce (sendbuf, recvbuf, count, MPI_INT, MPI_SUM, comm);
  sum = 0.;
  for (i = 0; i < commsize; ++i) 
    sum += i;
  chkbuf ("MPI_Allreduce", recvbuf, count, sum);

  gsbuf = (int *) malloc (commsize * count * sizeof (int));
  ret = MPI_Gather (sendbuf, count, MPI_INT,
		    gsbuf, count, MPI_INT,
		    0, comm);
  if (iam == 0) {
    val = 0;
    for (i = 0; i < commsize*count; ++i) {
      if (gsbuf[i] != val) {
	printf ("iam=%d MPI_Gather: bad gsbuf[%d]=%d != %d\n", iam, i, gsbuf[i], val);
	MPI_Abort (comm, -1);
      }
      if ((i+1) % count == 0)
	++val;
    }
  }

  ret = MPI_Scatter (gsbuf, count, MPI_INT,
		     recvbuf, count, MPI_INT,
		     0, comm);
  chkbuf ("MPI_Scatter", recvbuf, count, iam);

  atoabufsend = (int *) malloc (commsize * sizeof (int));
  atoabufrecv = (int *) malloc (commsize * sizeof (int));
  for (i = 0; i < commsize; ++i)
    atoabufsend[i] = i;

  ret = MPI_Alltoall (atoabufsend, 1, MPI_INT,
		      atoabufrecv, 1, MPI_INT,
		      comm);

  for (i = 0; i < commsize; ++i)
    if (atoabufrecv[i] != iam) {
      printf ("iam=%d MPI_Alltoall: bad atoabufrecv[%d]=%d != %d\n", iam, i, atoabufrecv[i], i);
      MPI_Abort (comm, -1);
    }

  ret = MPI_Reduce (sendbuf, recvbuf, count, MPI_INT, MPI_SUM, 0, comm);
  if (iam == 0) {
    sum = 0.;
    for (i = 0; i < commsize; ++i) 
      sum += i;
    chkbuf ("MPI_Reduce", recvbuf, count, sum);
  }

  ret = MPI_Finalize ();                       // Clean up MPI

  ret = GPTLstop ("total");
  ret = GPTLpr (iam);                          // Print the results
  return 0;
}

void chkbuf (const char *msg, int *recvbuf, const int count, const int source)
{
  int i;
  for (i = 0; i < count; ++i)
    if (recvbuf[i] != source) {
      printf ("iam=%d %s:bad recvbuf[%d]=%d != %d\n", iam, msg, i, recvbuf[i], source);
      MPI_Abort (comm, -1);
    }
}
