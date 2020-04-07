#ifndef GPTLMPI_H
#define GPTLMPI_H

#include <mpi.h>

extern "C" {
  // In pr_summary.cc:
  int GPTLpr_summary (MPI_Comm);
  int GPTLpr_summary_file (MPI_Comm, const char *);

  // In util.cc:
  int GPTLbarrier (MPI_Comm, const char *);
}
#endif
