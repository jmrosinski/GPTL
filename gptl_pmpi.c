/*
** $Id: gptl_pmpi.c,v 1.2 2009-12-24 22:40:52 rosinski Exp $
**
** Author: Jim Rosinski
**
** Wrappers to MPI routines
*/
 
#include "private.h"
#include "gptl.h"

#include <mpi.h>

int MPI_Send (void *buf, int count, MPI_Datatype datatype, int dest, int tag, 
	      MPI_Comm comm)
{
  int ret;
  int size;
  Timer *timer;

  (void) GPTLstart ("MPI_Send");
  ret = PMPI_Send (buf, count, datatype, dest, tag, comm);
  (void) GPTLstop ("MPI_Send");
  if ((timer = GPTLgetentry ("MPI_Send"))) {
    (void) PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Recv (void *buf, int count, MPI_Datatype datatype, int source, int tag, 
	      MPI_Comm comm, MPI_Status *status)
{
  int ret;
  int size;
  Timer *timer;

  (void) GPTLstart ("MPI_Recv");
  ret = PMPI_Recv (buf, count, datatype, source, tag, comm, status);
  (void) GPTLstop ("MPI_Recv");
  if (timer = GPTLgetentry ("MPI_Recv")) {
    (void) PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Isend (void *buf, int count, MPI_Datatype datatype, int dest, int tag, 
	       MPI_Comm comm, MPI_Request *request)
{
  int ret;
  int size;
  Timer *timer;

  (void) GPTLstart ("MPI_Isend");
  ret = PMPI_Isend (buf, count, datatype, dest, tag, comm, request);
  (void) GPTLstop ("MPI_Isend");
  if (timer = GPTLgetentry ("MPI_Isend")) {
    (void) PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Irecv (void *buf, int count, MPI_Datatype datatype, int source, int tag, 
	      MPI_Comm comm, MPI_Request *request)
{
  int ret;
  int size;
  Timer *timer;

  (void) GPTLstart ("MPI_Irecv");
  ret = PMPI_Irecv (buf, count, datatype, source, tag, comm, request);
  (void) GPTLstop ("MPI_Irecv");
  if (timer = GPTLgetentry ("MPI_Irecv")) {
    (void) PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Wait (MPI_Request *request, MPI_Status *status)
{
  int ret;

  (void) GPTLstart ("MPI_Wait");
  ret = PMPI_Wait (request, status);
  (void) GPTLstop ("MPI_Wait");
  return ret;
}

int MPI_Waitall(int count, 
		MPI_Request array_of_requests[], 
		MPI_Status array_of_statuses[])
{
  int ret;

  (void) GPTLstart ("MPI_Waitall");
  ret = PMPI_Waitall (count, array_of_requests, array_of_statuses);
  (void) GPTLstop ("MPI_Waitall");
  return ret;
}

int MPI_Barrier (MPI_Comm comm)
{
  int ret;

  (void) GPTLstart ("MPI_Barrier");
  ret = PMPI_Barrier (comm);
  (void) GPTLstop ("MPI_Barrier");
  return ret;
}

int MPI_Bcast (void *buffer, int count, MPI_Datatype datatype, int root, 
               MPI_Comm comm )
{
  int ret;
  int iam;
  int size;
  Timer *timer;

  (void) GPTLstart ("MPI_Bcast");
  ret = PMPI_Bcast (buffer, count, datatype, root, comm);
  (void) GPTLstop ("MPI_Bcast");
  if (timer = GPTLgetentry ("MPI_Bcast")) {
    (void) PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Allreduce (void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, 
		   MPI_Op op, MPI_Comm comm)
{
  int ret;
  int size;
  Timer *timer;

  (void) GPTLstart ("MPI_Allreduce");
  ret = PMPI_Allreduce (sendbuf, recvbuf, count, datatype, op, comm);
  (void) GPTLstop ("MPI_Allreduce");
  if (timer = GPTLgetentry ("MPI_Allreduce")) {
    (void) PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Gather (void *sendbuf, int sendcnt, MPI_Datatype sendtype, 
                void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                int root, MPI_Comm comm)
{
  int ret;
  int iam;
  int sendsize, recvsize;
  int commsize;
  Timer *timer;

  (void) GPTLstart ("MPI_Gather");
  ret = PMPI_Gather (sendbuf, sendcnt, sendtype, 
		     recvbuf, recvcount, recvtype, root, comm);
  (void) GPTLstop ("MPI_Gather");
  if (timer = GPTLgetentry ("MPI_Gather")) {
    (void) PMPI_Comm_rank (comm, &iam);
    (void) PMPI_Comm_size (comm, &commsize);
    (void) PMPI_Type_size (sendtype, &sendsize);
    (void) PMPI_Type_size (recvtype, &recvsize);

    if (iam == root) {

      /* Use size-1 to exclude root sending to himself */

      timer->nbytes += (double) recvcount * recvsize * (commsize - 1);
    } else {
      timer->nbytes += (double) sendcnt * sendsize;
    }
  }
  return ret;
}
