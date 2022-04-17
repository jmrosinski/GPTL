/*
** util.c
**
** Author: Jim Rosinski
**
** Utility functions, mostly for printing error and warning messages
*/

#include "config.h"   // Must be first include

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

#include "gptl.h"
#include "private.h"
#include "once.h"
#include "util.h"

static inline bool doprint(void);

namespace util {
  bool abort_on_error = false;            // flag says to abort on any error
  unsigned long max_errors = 10;          // max number of error print msgs
  volatile unsigned long num_errors = 0;  // number of times GPTLerror was called
  unsigned long max_warn = 10;            // max number of warning messages
  volatile unsigned long num_warn = 0;    // number of times GPTLwarn was called
  
  /*
  ** error: error return routine to print a message and return a failure value.
  **
  ** Input arguments:
  **   fmt: format string
  **   variable list of additional arguments for vfprintf
  **
  ** Return value: -1 (failure)
  */
  int error (const char *fmt, ...)
  {
    va_list args;
    
    if (doprint()) {
      va_start (args, fmt);
      if (fmt != NULL && util::num_errors < util::max_errors) {
	(void) fprintf (stderr, "GPTL error:");
	(void) vfprintf (stderr, fmt, args);
	if (util::num_errors >= util::max_errors)
	  (void) fprintf (stderr, "Truncating further error print now after %lu msgs",
			  util::num_errors);
      }
      va_end (args);
    }
    
    if (util::abort_on_error) {
      (void) fprintf (stderr, "GPTL error:");
      exit (-1);
    }
#pragma omp critical
    {
      if (util::num_errors < ULONG_MAX)
	++util::num_errors;
    }
    return -1;
  }
  
  /*
  ** warn: print a warning message
  **
  ** Input arguments:
  **   fmt: format string
  **   variable list of additional arguments for vfprintf
  */
  void warn (const char *fmt, ...)
  {
    va_list args;
    
    if (doprint()) {
      va_start (args, fmt);
      if (fmt != NULL && num_warn < max_warn) {
	(void) fprintf (stderr, "GPTL warning:");
	(void) vfprintf (stderr, fmt, args);
	if (util::num_warn >= util::max_warn)
	  (void) fprintf (stderr, "Truncating further warning print now after %lu msgs",
			  util::num_warn);
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
  void note (const char *fmt, ...)
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
  
  /*
  ** allocate: wrapper utility for malloc
  **
  ** Input arguments:
  **   nbytes: size to allocate
  **
  ** Return value: pointer to the new space (or NULL)
  */
  void *allocate (const int nbytes, const char *caller)
  {
    void *ptr = NULL;
    
    if ( nbytes <= 0 || ! (ptr = malloc (nbytes)))
      (void) error ("allocate from %s: malloc failed for %d bytes\n", nbytes, caller);
    
    return ptr;
  }  

  // set_abort_on_error: Set abort_on_error flag
  void set_abort_on_error (bool val) {util::abort_on_error = val;}

  // GPTLreset_errors: reset error state to no errors
  void reset_errors (void) {util::num_errors = 0;}
}

// Local functions start here
static inline bool doprint()
{
#ifdef HAVE_LIBMPI
  static int world_iam = 0;  // MPI rank
  int flag;
  int ret;
  static bool check_mpi_init = true;

  // Only need to change GPTLworld_iam from 0 in MPI jobs, when flag set to only print 
  // from rank 0, and when MPI_Initialize hasn't already been invoked
  if (onlyprint_rank0 && check_mpi_init) {
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

// User-accessible functions start here

// GPTLnum_errors: User-visible routine returns number of times GPTLerror() called
int GPTLnum_errors (void) {return util::num_errors;}

// GPTLnum_warn: User-visible routine returns number of times GPTLerror() called
int GPTLnum_warn (void) {return util::num_warn;}

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

  ret = gptlmain::start (name);
  if ((ret = MPI_Barrier (comm)) != MPI_SUCCESS)
    return error ("%s: Bad return from MPI_Barrier=%d", thisfunc, ret);
  ret = gptlmain::stop (name);
  return ret;
}
#endif    // HAVE_LIBMPI
