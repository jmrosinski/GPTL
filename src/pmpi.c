/*
** pmpi.c
**
** Author: Jim Rosinski
**
** Wrappers to MPI routines
*/
 
#include "config.h" /* Must be first include. */
#include "private.h"
#include "gptl.h"
#include <mpi.h>

static bool sync_mpi = false;

#ifdef __cplusplus
extern "C" {
#endif

int GPTLpmpi_setoption (const int option,
			const int val)
{
  int retval;

  switch (option) {
  case GPTLsync_mpi:
    sync_mpi = (bool) val;
    retval = 0;
    break;
  default:
    retval = 1;
  }
  return retval;
}

int MPI_Send (const void *buf, int count, MPI_Datatype datatype, int dest, int tag, 
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
    /* Ignore status */
    ignoreret = PMPI_Probe (source, tag, comm, status);
    ignoreret = GPTLstop ("sync_Recv");
  }
    
  ignoreret = GPTLstart ("MPI_Recv");
  ret = PMPI_Recv (buf, count, datatype, source, tag, comm, status);
  ignoreret = GPTLstop ("MPI_Recv");
  if ((timer = GPTLgetentry ("MPI_Recv"))) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Sendrecv (const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, 
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
  if ((timer = GPTLgetentry ("MPI_Sendrecv"))) {
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);

    timer->nbytes += ((double) recvcount * recvsize) + ((double) sendcount * sendsize);
  }
  return ret;
}

int MPI_Isend (const void *buf, int count, MPI_Datatype datatype, int dest, int tag, 
	       MPI_Comm comm, MPI_Request *request)
{
  int ret;
  int ignoreret;
  int size;
  Timer *timer;

  ignoreret = GPTLstart ("MPI_Isend");
  ret = PMPI_Isend (buf, count, datatype, dest, tag, comm, request);
  ignoreret = GPTLstop ("MPI_Isend");
  if ((timer = GPTLgetentry ("MPI_Isend"))) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Issend (const void *buf, int count, MPI_Datatype datatype, int dest, int tag, 
		MPI_Comm comm, MPI_Request *request)
{
  int ret;
  int ignoreret;
  int size;
  Timer *timer;

  ignoreret = GPTLstart ("MPI_Issend");
  ret = PMPI_Issend (buf, count, datatype, dest, tag, comm, request);
  ignoreret = GPTLstop ("MPI_Issend");
  if ((timer = GPTLgetentry ("MPI_Issend"))) {
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
  if ((timer = GPTLgetentry ("MPI_Irecv"))) {
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
  if ((timer = GPTLgetentry ("MPI_Bcast"))) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Allreduce (const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, 
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
  if ((timer = GPTLgetentry ("MPI_Allreduce"))) {
    ignoreret = PMPI_Type_size (datatype, &size);
    /* Estimate size as 1 send plus 1 recv */
    timer->nbytes += 2.*((double) count) * size;
  }
  return ret;
}

int MPI_Gather (const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
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

  if ((timer = GPTLgetentry ("MPI_Gather"))) {
    ignoreret = PMPI_Comm_rank (comm, &iam);
    ignoreret = PMPI_Comm_size (comm, &commsize);
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);
    timer->nbytes += (double) sendcount * sendsize;
    if (iam == root) {
      timer->nbytes += (double) recvcount * recvsize * (commsize-1);
    }
  }
  return ret;
}

int MPI_Gatherv (const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
		 void *recvbuf, const int *recvcounts, const int *displs, 
                 MPI_Datatype recvtype, int root, MPI_Comm comm )
{
  int ret;
  int iam;
  int i;
  int sendsize, recvsize;
  int commsize;
  int ignoreret;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Gatherv");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Gatherv");
  }
    
  ignoreret = GPTLstart ("MPI_Gatherv");
  ret = PMPI_Gatherv (sendbuf, sendcount, sendtype, 
		      recvbuf, recvcounts, displs, 
		      recvtype, root, comm);
  ignoreret = GPTLstop ("MPI_Gatherv");

  if ((timer = GPTLgetentry ("MPI_Gatherv"))) {
    ignoreret = PMPI_Comm_rank (comm, &iam);
    ignoreret = PMPI_Comm_size (comm, &commsize);
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);
    if (iam == root) {
      for (i = 0; i < commsize; ++i)
	if (i != iam)
	  timer->nbytes += (double) recvcounts[i] * recvsize;
    } else {
      timer->nbytes += (double) sendcount * sendsize;
    }
  }
  return ret;
}

int MPI_Scatter (const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
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
  if ((timer = GPTLgetentry ("MPI_Scatter"))) {
    ignoreret = PMPI_Comm_rank (comm, &iam);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);
    timer->nbytes += (double) recvcount * recvsize;
    if (iam == root) {
      ignoreret = PMPI_Type_size (sendtype, &sendsize);
      timer->nbytes += (double) sendcount * sendsize;
    }
  }
  return ret;
}

int MPI_Alltoall (const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                  void *recvbuf, int recvcount, MPI_Datatype recvtype, 
		  MPI_Comm comm)
{
  int ret;
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
  if ((timer = GPTLgetentry ("MPI_Alltoall"))) {
    ignoreret = PMPI_Comm_size (comm, &commsize);
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);

    timer->nbytes += ((double) sendcount * sendsize * (commsize-1)) + 
                     ((double) recvcount * recvsize * (commsize-1));
  }
  return ret;
}

int MPI_Reduce (const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, 
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
  if ((timer = GPTLgetentry ("MPI_Reduce"))) {
    ignoreret = PMPI_Type_size (datatype, &size);
    /* Estimate byte count as 1 send */
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Allgather (const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                   void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                   MPI_Comm comm)
{
  int ret;
  int sendsize, recvsize;
  int commsize;
  int ignoreret;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Allgather");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Allgather");
  }
    
  ignoreret = GPTLstart ("MPI_Allgather");
  ret = PMPI_Allgather (sendbuf, sendcount, sendtype, 
			recvbuf, recvcount, recvtype, comm);
  ignoreret = GPTLstop ("MPI_Allgather");

  if ((timer = GPTLgetentry ("MPI_Allgather"))) {
    ignoreret = PMPI_Comm_size (comm, &commsize);
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);
    timer->nbytes += (double) sendcount * sendsize * (commsize-1)+ 
                     (double) recvcount * recvsize * (commsize-1);
  }
  return ret;
}

int MPI_Allgatherv (const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                    void *recvbuf, const int *recvcounts, const int *displs, 
                    MPI_Datatype recvtype, MPI_Comm comm )
{
  int ret;
  int iam;
  int i;
  int sendsize, recvsize;
  int commsize;
  int ignoreret;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Allgatherv");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Allgatherv");
  }
    
  ignoreret = GPTLstart ("MPI_Allgatherv");
  ret = PMPI_Allgatherv (sendbuf, sendcount, sendtype, 
			 recvbuf, recvcounts, displs, 
			 recvtype, comm);
  ignoreret = GPTLstop ("MPI_Allgatherv");

  if ((timer = GPTLgetentry ("MPI_Allgatherv"))) {
    ignoreret = PMPI_Comm_rank (comm, &iam);
    ignoreret = PMPI_Comm_size (comm, &commsize);
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);
    timer->nbytes += (double) sendcount * sendsize * (commsize-1);
    for (i = 0; i < commsize; ++i)
      if (i != iam)
	timer->nbytes += (double) recvcounts[i] * recvsize;
  }
  return ret;
}

int MPI_Iprobe (int source, int tag, MPI_Comm comm, int *flag,
		MPI_Status *status)
{
  int ret;
  int ignoreret;

  ignoreret = GPTLstart ("MPI_Iprobe");
  ret = PMPI_Iprobe (source, tag, comm, flag, status);
  ignoreret = GPTLstop ("MPI_Iprobe");
  return ret;
}

int MPI_Probe (int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  int ret;
  int ignoreret;

  ignoreret = GPTLstart ("MPI_Probe");
  ret = PMPI_Probe (source, tag, comm, status);
  ignoreret = GPTLstop ("MPI_Probe");
  return ret;
}

int MPI_Ssend (const void *buf, int count, MPI_Datatype datatype,
	       int dest, int tag, MPI_Comm comm)
{
  int ret;
  int ignoreret;
  int size;
  Timer *timer;

  ignoreret = GPTLstart ("MPI_Ssend");
  ret = PMPI_Ssend (buf, count, datatype, dest, tag, comm);
  ignoreret = GPTLstop ("MPI_Ssend");
  if ((timer = GPTLgetentry ("MPI_Ssend"))) {
    ignoreret = PMPI_Type_size (datatype, &size);
    timer->nbytes += ((double) count) * size;
  }
  return ret;
}

int MPI_Alltoallv (const void *sendbuf, const int *sendcounts, const int *sdispls,
		   MPI_Datatype sendtype, void *recvbuf, const int *recvcounts,
		   const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  int ret;
  int iam;
  int i;
  int sendsize, recvsize;
  int commsize;
  int ignoreret;
  Timer *timer;
  
  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Alltoallv");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Alltoallv");
  }
  
  ignoreret = GPTLstart ("MPI_Alltoallv");
  ret = PMPI_Alltoallv (sendbuf, sendcounts, sdispls,
			sendtype, recvbuf, recvcounts,
			rdispls, recvtype, comm);
  
  ignoreret = GPTLstop ("MPI_Alltoallv");
  if ((timer = GPTLgetentry ("MPI_Alltoallv"))) {
    ignoreret = PMPI_Comm_rank (comm, &iam);
    ignoreret = PMPI_Comm_size (comm, &commsize);
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);
    for (i = 0; i < commsize; ++i) {
      if (i != iam) {
	timer->nbytes += (double) sendcounts[i] * sendsize;
	timer->nbytes += (double) recvcounts[i] * recvsize;
      }
    }
  }
  return ret;
}

int MPI_Scatterv (const void *sendbuf, const int *sendcounts, const int *displs,
		  MPI_Datatype sendtype, void *recvbuf, int recvcount,
		  MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  int ret;
  int iam;
  int i;
  int sendsize, recvsize;
  int commsize;
  int ignoreret;
  Timer *timer;

  if (sync_mpi) {
    ignoreret = GPTLstart ("sync_Scatterv");
    ignoreret = PMPI_Barrier (comm);
    ignoreret = GPTLstop ("sync_Scatterv");
  }
    
  ignoreret = GPTLstart ("MPI_Scatterv");
  ret = PMPI_Scatterv (sendbuf, sendcounts, displs,
		       sendtype, recvbuf, recvcount, 
		       recvtype, root, comm);
  ignoreret = GPTLstop ("MPI_Scatterv");
  if ((timer = GPTLgetentry ("MPI_Scatterv"))) {
    ignoreret = PMPI_Comm_rank (comm, &iam);
    ignoreret = PMPI_Comm_size (comm, &commsize);
    ignoreret = PMPI_Type_size (sendtype, &sendsize);
    ignoreret = PMPI_Type_size (recvtype, &recvsize);
    timer->nbytes += (double) recvcount * recvsize;
    if (iam == root) {
      for (i = 0; i < commsize; ++i)
	if (i != iam)
	  timer->nbytes += (double) sendcounts[i] * sendsize;
    } else {
      timer->nbytes += (double) recvcount * recvsize;
    }
  }
  return ret;
}

int MPI_Test (MPI_Request *request, int *flag, MPI_Status *status)
{
  int ret;
  int ignoreret;

  ignoreret = GPTLstart ("MPI_Test");
  ret = PMPI_Test (request, flag, status);
  ignoreret = GPTLstop ("MPI_Test");
  return ret;
}
#endif

#ifdef __cplusplus
}
#endif
