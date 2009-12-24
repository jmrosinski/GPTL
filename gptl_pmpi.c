/*
** $Id: gptl_pmpi.c,v 1.1 2009-12-24 21:26:39 rosinski Exp $
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
  Timer *timer;

  (void) GPTLstart ("MPI_Send");
  ret = PMPI_Send (buf, count, datatype, dest, tag, comm);
  (void) GPTLstop ("MPI_Send");
  if (timer = GPTLgetentry ("MPI_Send")) {
    timer->things += (double) count;
  }
  return ret;
}

int MPI_Recv (void *buf, int count, MPI_Datatype datatype, int source, int tag, 
	      MPI_Comm comm, MPI_Status *status)
{
  int ret;
  Timer *timer;

  (void) GPTLstart ("MPI_Recv");
  ret = PMPI_Recv (buf, count, datatype, source, tag, comm, status);
  (void) GPTLstop ("MPI_Recv");
  if (timer = GPTLgetentry ("MPI_Recv")) {
    timer->things += (double) count;
  }
  return ret;
}

int MPI_Isend (void *buf, int count, MPI_Datatype datatype, int dest, int tag, 
	       MPI_Comm comm, MPI_Request *request)
{
  int ret;
  Timer *timer;

  (void) GPTLstart ("MPI_Isend");
  ret = PMPI_Isend (buf, count, datatype, dest, tag, comm, request);
  (void) GPTLstop ("MPI_Isend");
  if (timer = GPTLgetentry ("MPI_Isend")) {
    timer->things += (double) count;
  }
  return ret;
}

int MPI_Irecv (void *buf, int count, MPI_Datatype datatype, int source, int tag, 
	      MPI_Comm comm, MPI_Request *request)
{
  int ret;
  Timer *timer;

  (void) GPTLstart ("MPI_Irecv");
  ret = PMPI_Irecv (buf, count, datatype, source, tag, comm, request);
  (void) GPTLstop ("MPI_Irecv");
  if (timer = GPTLgetentry ("MPI_Irecv")) {
    timer->things += (double) count;
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
  ret = MPI_Barrier (comm);
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
  (void) GPTLstop ("MPI_Bcast");
  if (timer = GPTLgetentry ("MPI_Bcast")) {
    (void) MPI_Comm_rank (comm, &iam);
    (void) MPI_Comm_size (comm, &size);
    if (iam == root) {

      /* Use size-1 to exclude root sending to himself */

      timer->things += (double) (count - 1) * size;
    } else {
      timer->things += (double) size;
    }
  }
}

int MPI_Allreduce (void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, 
		   MPI_Op op, MPI_Comm comm)
{
  int ret;
  Timer *timer;

  (void) GPTLstart ("MPI_Allreduce");
  ret = PMPI_Allreduce (sendbuf, recvbuf, count, datatype, op, comm);
  (void) GPTLstop ("MPI_Allreduce");
  if (timer = GPTLgetentry ("MPI_Allreduce")) {
    timer->things += (double) count;
  }
  return ret;
}

int MPI_Gather (void *sendbuf, int sendcnt, MPI_Datatype sendtype, 
                void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                int root, MPI_Comm comm)
{
  int ret;
  int iam;
  int size;
  Timer *timer;

  (void) GPTLstart ("MPI_Gather");
  ret = PMPI_Gather (sendbuf, sendcnt, sendtype, 
		     recvbuf, recvcount, recvtype, root, comm);
  (void) GPTLstop ("MPI_Gather");
  if (timer = GPTLgetentry ("MPI_Gather")) {
    (void) MPI_Comm_rank (comm, &iam);
    (void) MPI_Comm_size (comm, &size);
    if (iam == root) {

      /* Use size-1 to exclude root sending to himself */

      timer->things += (double) recvcount * (size - 1);
    } else {
      timer->things += (double) sendcnt;
    }
  }
  return ret;
}
