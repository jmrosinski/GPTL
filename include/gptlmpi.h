#ifndef GPTLMPI_H
#define GPTLMPI_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int GPTLpr_summary (MPI_Comm);
extern int GPTLpr_summary_file (MPI_Comm, const char *);
extern int GPTLbarrier (MPI_Comm, const char *);

#ifdef __cplusplus
}
#endif
#endif
