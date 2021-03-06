/*
** util.c
**
** Author: Jim Rosinski
**
** Utility functions, mostly for printing error and warning messages
*/

#include "config.h"   // Must be first include
#include "private.h"
#include "gptl.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#ifdef UNDERLYING_OPENMP
#include <omp.h>
#endif
#ifdef HAVE_LIBMPI
#include <mpi.h>
#endif

static bool abort_on_error = false;            // flag says to abort on any error
static unsigned long max_errors = 10;          // max number of error print msgs
static volatile unsigned long num_errors = 0;  // number of times GPTLerror was called
static unsigned long max_warn = 10;            // max number of warning messages
static volatile unsigned long num_warn = 0;    // number of times GPTLwarn was called

#ifdef __cplusplus
extern "C" {
#endif

static inline bool doprint(void);

/*
** GPTLerror: error return routine to print a message and return a failure value.
**
** Input arguments:
**   fmt: format string
**   variable list of additional arguments for vfprintf
**
** Return value: -1 (failure)
*/
int GPTLerror (const char *fmt, ...)
{
  va_list args;
  
  if (doprint()) {
    va_start (args, fmt);
    if (fmt != NULL && num_errors < max_errors) {
      (void) fprintf (stderr, "GPTL error:");
      (void) vfprintf (stderr, fmt, args);
      if (num_errors >= max_errors)
	(void) fprintf (stderr, "Truncating further error print now after %lu msgs", num_errors);
    }
    va_end (args);
  }
  
  if (abort_on_error) {
    (void) fprintf (stderr, "GPTL error:");
    exit (-1);
  }
#pragma omp critical
  {
    if (num_errors < ULONG_MAX)
      ++num_errors;
  }
  return -1;
}

/*
** GPTLwarn: print a warning message
**
** Input arguments:
**   fmt: format string
**   variable list of additional arguments for vfprintf
*/
void GPTLwarn (const char *fmt, ...)
{
  va_list args;
  
  if (doprint()) {
    va_start (args, fmt);
    if (fmt != NULL && num_warn < max_warn) {
      (void) fprintf (stderr, "GPTL warning:");
      (void) vfprintf (stderr, fmt, args);
      if (num_warn >= max_warn)
	(void) fprintf (stderr, "Truncating further warning print now after %lu msgs", num_warn);
    }
    va_end (args);
  }
#pragma omp critical
  {
    if (num_warn < ULONG_MAX)
      ++num_warn;
  }
}

/*
** GPTLnote: print a note
**
** Input arguments:
**   fmt: format string
**   variable list of additional arguments for vfprintf
*/
void GPTLnote (const char *fmt, ...)
{
  va_list args;
  
  if (doprint()) {
    va_start (args, fmt);
    if (fmt != NULL) {
      (void) fprintf (stderr, "GPTL note:");
      (void) vfprintf (stderr, fmt, args);
    }
    va_end (args);
  }
}

static inline bool doprint()
{
#ifdef HAVE_LIBMPI
  static int world_iam = 0;  // MPI rank
  int flag;
  int ret;
  static bool check_mpi_init = true;

  // Only need to change GPTLworld_iam from 0 in MPI jobs, when flag set to only print 
  // from rank 0, and when MPI_Initialize hasn't already been invoked
  if (GPTLonlypr_rank0 && check_mpi_init) {
    ret = MPI_Initialized (&flag);
    if (flag) {
      check_mpi_init = false;
      ret = MPI_Comm_rank (MPI_COMM_WORLD, &world_iam);
    }
  }
  return world_iam == 0;
#else
  return true;  // Always return true when no MPI
#endif
}

// set_abort_on_error: Set abort_on_error flag
void GPTLset_abort_on_error (bool val) {abort_on_error = val;}

// GPTLreset_errors: reset error state to no errors
void GPTLreset_errors (void) {num_errors = 0;}

// GPTLnum_errors: User-visible routine returns number of times GPTLerror() called
int GPTLnum_errors (void) {return num_errors;}

// GPTLnum_errors: User-visible routine returns number of times GPTLerror() called
int GPTLnum_warn (void) {return num_warn;}

/*
** GPTLallocate: wrapper utility for malloc
**
** Input arguments:
**   nbytes: size to allocate
**
** Return value: pointer to the new space (or NULL)
*/
void *GPTLallocate (const int nbytes, const char *caller)
{
  void *ptr = NULL;

  if ( nbytes <= 0 || ! (ptr = malloc (nbytes)))
    (void) GPTLerror ("GPTLallocate from %s: malloc failed for %d bytes\n", nbytes, caller);

  return ptr;
}

#ifdef HAVE_LIBMPI
/* 
** GPTLbarrier: When MPI enabled, set and time an MPI barrier
**
** Input arguments:
**   comm: commuicator (e.g. MPI_COMM_WORLD). If zero, use MPI_COMM_WORLD
**   name: region name
**
** Return value: 0 (success)
*/
int GPTLbarrier (MPI_Comm comm, const char *name)
{
  int ret;
  static const char *thisfunc = "GPTLbarrier";

  ret = GPTLstart (name);
  if ((ret = MPI_Barrier (comm)) != MPI_SUCCESS)
    return GPTLerror ("%s: Bad return from MPI_Barrier=%d", thisfunc, ret);
  ret = GPTLstop (name);
  return ret;
}
#endif    // HAVE_LIBMPI

#ifdef __cplusplus
}
#endif
