/*
** $Id: gptl_pmpi.c,v 1.6 2009-12-26 19:27:22 rosinski Exp $
**
** Author: Jim Rosinski
**
** Wrappers to MPI routines
*/
 
#include "private.h"
#include "gptl.h"

#include <mpi.h>

static bool sync_mpi = false;

int GPTLpmpi_setoption (const int option,
			const int val)
{
  switch (option) {
  case GPTLsync_mpi:
    sync_mpi = (bool) val;
    return 0;
  default:
    return 1;
  }
  return 0;
}

/*
** Additions to MPI_Init: Initialize GPTL if this hasn't already been done.
** Start a timer which will be stopped in MPI_Finalize.
*/

int MPI_Init (int *argc, char ***argv)
{
  int ret;
  int ignoreret;

  ret = PMPI_Init (argc, argv);
  if ( ! GPTLis_initialized ())
    ignoreret = GPTLinitialize ();

  ignoreret = GPTLstart ("MPI_Init_thru_Finalize");

  return ret;
}

int MPI_Send (void *buf, int count, MPI_Datatype datatype, int dest, int tag, 
	      MPI_Comm comm)
{
  int ret;
  int size;
  int ignoreret;
  Timer *timer;

  ignoreret = GPTLstart ("MPI_Send");
  ret = PMPI_Send (buf, count, datatype, dest, tag, comm);
  ignoreret = GPTLstop ("MPI_Send");
  if ((timer = GPTLgetentry ("MPI_Send"))) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Recv (void *buf, int count, MPI_Datatype datatype, int source, int tag, 
	      MPI_Comm comm, MPI_Status *status)
{
  int ret;
  int ignoreret;
  int size;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Recv");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Recv");
  }
    
  ignoreret = GPTLstart ("MPI_Recv");
  ret = PMPI_Recv (buf, count, datatype, source, tag, comm, status);
  ignoreret = GPTLstop ("MPI_Recv");
  if (timer = GPTLgetentry ("MPI_Recv")) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Sendrecv (void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, 
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, 
		  MPI_Comm comm, MPI_Status *status )
{
  int ret;
  int ignoreret;
  int sendsize, recvsize;
  Timer *timer;

  ignoreret = GPTLstart ("MPI_Sendrecv");
  ret = PMPI_Sendrecv (sendbuf, sendcount, sendtype, dest, sendtag, 
		       recvbuf, recvcount, recvtype, source, recvtag, comm, status);
  ignoreret = GPTLstop ("MPI_Sendrecv");
  if (timer = GPTLgetentry ("MPI_Sendrecv")) {
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);

    timer->nbytes += ((double) recvcount * recvsize) + ((double) sendcount * sendsize);
  }
  return ret;
}

int MPI_Isend (void *buf, int count, MPI_Datatype datatype, int dest, int tag, 
	       MPI_Comm comm, MPI_Request *request)
{
  int ret;
  int ignoreret;
  int size;
  Timer *timer;

  ignoreret = GPTLstart ("MPI_Isend");
  ret = PMPI_Isend (buf, count, datatype, dest, tag, comm, request);
  ignoreret = GPTLstop ("MPI_Isend");
  if (timer = GPTLgetentry ("MPI_Isend")) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Irecv (void *buf, int count, MPI_Datatype datatype, int source, int tag, 
	      MPI_Comm comm, MPI_Request *request)
{
  int ret;
  int ignoreret;
  int size;
  Timer *timer;

  ignoreret = GPTLstart ("MPI_Irecv");
  ret = PMPI_Irecv (buf, count, datatype, source, tag, comm, request);
  ignoreret = GPTLstop ("MPI_Irecv");
  if (timer = GPTLgetentry ("MPI_Irecv")) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Wait (MPI_Request *request, MPI_Status *status)
{
  int ret;
  int ignoreret;

  ignoreret = GPTLstart ("MPI_Wait");
  ret = PMPI_Wait (request, status);
  ignoreret = GPTLstop ("MPI_Wait");
  return ret;
}

int MPI_Waitall(int count, 
		MPI_Request array_of_requests[], 
		MPI_Status array_of_statuses[])
{
  int ret;
  int ignoreret;

  ignoreret = GPTLstart ("MPI_Waitall");
  ret = PMPI_Waitall (count, array_of_requests, array_of_statuses);
  ignoreret = GPTLstop ("MPI_Waitall");
  return ret;
}

int MPI_Barrier (MPI_Comm comm)
{
  int ret;
  int ignoreret;

  ignoreret = GPTLstart ("MPI_Barrier");
  ret = PMPI_Barrier (comm);
  ignoreret = GPTLstop ("MPI_Barrier");
  return ret;
}

int MPI_Bcast (void *buffer, int count, MPI_Datatype datatype, int root, 
               MPI_Comm comm )
{
  int ret;
  int ignoreret;
  int size;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Bcast");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Bcast");
  }
    
  ignoreret = GPTLstart ("MPI_Bcast");
  ret = PMPI_Bcast (buffer, count, datatype, root, comm);
  ignoreret = GPTLstop ("MPI_Bcast");
  if (timer = GPTLgetentry ("MPI_Bcast")) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Allreduce (void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, 
		   MPI_Op op, MPI_Comm comm)
{
  int ret;
  int ignoreret;
  int size;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Allreduce");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Allreduce");
  }
    
  ignoreret = GPTLstart ("MPI_Allreduce");
  ret = PMPI_Allreduce (sendbuf, recvbuf, count, datatype, op, comm);
  ignoreret = GPTLstop ("MPI_Allreduce");
  if (timer = GPTLgetentry ("MPI_Allreduce")) {
    ignoreret = PMPI_Type_size (datatype, &size);
    /* Estimate size as 2*count*size */
    timer->nbytes += 2*((double) count) * size;
  }
  return ret;
}

int MPI_Gather (void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                int root, MPI_Comm comm)
{
  int ret;
  int iam;
  int sendsize, recvsize;
  int commsize;
  int ignoreret;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Gather");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Gather");
  }
    
  ignoreret = GPTLstart ("MPI_Gather");
  ret = PMPI_Gather (sendbuf, sendcount, sendtype, 
		     recvbuf, recvcount, recvtype, root, comm);
  ignoreret = GPTLstop ("MPI_Gather");

  if (timer = GPTLgetentry ("MPI_Gather")) {
    ignoreret = PMPI_Comm_rank (comm, &iam);
    ignoreret = PMPI_Comm_size (comm, &commsize);
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);

    if (iam == root) {

      /* Use size-1 to exclude root sending to himself */

      timer->nbytes += (double) recvcount * recvsize * (commsize - 1);
    } else {
      timer->nbytes += (double) sendcount * sendsize;
    }
  }
  return ret;
}

int MPI_Scatter (void *sendbuf, int sendcount, MPI_Datatype sendtype, 
		 void *recvbuf, int recvcount, MPI_Datatype recvtype, 
		 int root, MPI_Comm comm)
{
  int ret;
  int iam;
  int sendsize, recvsize;
  int ignoreret;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Scatter");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Scatter");
  }
    
  ignoreret = GPTLstart ("MPI_Scatter");
  ret = PMPI_Scatter (sendbuf, sendcount, sendtype, 
		      recvbuf, recvcount, recvtype, root, comm);
  ignoreret = GPTLstop ("MPI_Scatter");
  if (timer = GPTLgetentry ("MPI_Scatter")) {
    ignoreret = PMPI_Comm_rank (comm, &iam);
    if (iam == root) {

      /* Use size-1 to exclude root sending to himself */

      ignoreret = PMPI_Type_size (sendtype, &sendsize);
      timer->nbytes += (double) sendcount * sendsize;
    } else {
      ignoreret = PMPI_Type_size (recvtype, &recvsize);
      timer->nbytes += (double) recvcount * recvsize;
    }
  }
  return ret;
}

int MPI_Alltoall (void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, 
		  MPI_Comm comm)
{
  int ret;
  int iam;
  int sendsize, recvsize;
  int commsize;
  int ignoreret;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Alltoall");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Alltoall");
  }
    
  ignoreret = GPTLstart ("MPI_Alltoall");
  ret = PMPI_Alltoall (sendbuf, sendcount, sendtype, 
		       recvbuf, recvcount, recvtype, comm);
  ignoreret = GPTLstop ("MPI_Alltoall");
  if (timer = GPTLgetentry ("MPI_Alltoall")) {
    ignoreret = PMPI_Comm_size (comm, &commsize);
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);

    timer->nbytes += ((double) sendcount * sendsize) + 
                     ((double) recvcount * recvsize * commsize);
  }
  return ret;
}

int MPI_Reduce (void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, 
                MPI_Op op, int root, MPI_Comm comm )
{
  int ret;
  int size;
  int ignoreret;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Reduce");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Reduce");
  }
    
  ignoreret = GPTLstart ("MPI_Reduce");
  ret = PMPI_Reduce (sendbuf, recvbuf, count, datatype, op, root, comm);
  ignoreret = GPTLstop ("MPI_Reduce");
  if (timer = GPTLgetentry ("MPI_Reduce")) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

/*
** Additions to MPI_Finalize: Stop the timer started in MPI_Init, and
** call GPTLpr() if it hasn't already been called.
*/

int MPI_Finalize (void)
{
  int ret, ignoreret;
  int iam;

  ignoreret = GPTLstop ("MPI_Init_thru_Finalize");

  if ( ! GPTLpr_has_been_called ()) {
    PMPI_Comm_rank (MPI_COMM_WORLD, &iam);
    ignoreret = GPTLpr (iam);
  }

  ret = PMPI_Finalize();
  return ret;
}
